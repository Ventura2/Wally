from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wally.planner.rollout import LatentRollout
from wally.planner.trm_head import TRMHead


class EncodedChunkDataset(Dataset):
    def __init__(self, encoded: np.ndarray, chunk_size: int = 64) -> None:
        self.encoded = encoded
        self.chunk_size = chunk_size
        self.n_chunks = len(encoded) // chunk_size

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.chunk_size
        end = start + self.chunk_size
        return torch.from_numpy(self.encoded[start:end])


def encode_shards(
    rollout: LatentRollout,
    shard_dir: Path,
    max_chunks: int,
    chunk_size: int = 64,
) -> np.ndarray:
    inner = rollout._model._model
    inner.eval()
    device = next(inner.parameters()).device
    encoded_chunks: list[np.ndarray] = []
    shard_paths = sorted(shard_dir.glob("shard_*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.tar in {shard_dir}")
    for shard_path in shard_paths:
        with tarfile.open(shard_path, "r") as tf:
            for member in tf:
                if not member.name.endswith(".npz"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = np.load(f, allow_pickle=True)
                frames = data["frames"]
                ft = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
                ft = ft.to(device)
                with torch.no_grad():
                    z = inner.encoder(ft)
                if z.dim() == 4:
                    z = z.mean(dim=(2, 3))
                elif z.dim() == 3:
                    z = z.mean(dim=1)
                encoded_chunks.append(z.cpu().numpy().astype(np.float32))
                if len(encoded_chunks) >= max_chunks:
                    return np.concatenate(encoded_chunks, axis=0)
    if not encoded_chunks:
        raise RuntimeError("No chunks encoded")
    return np.concatenate(encoded_chunks, axis=0)


def sample_pairs(
    encoded: np.ndarray,
    chunk_size: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_chunks = len(encoded) // chunk_size
    max_dt = chunk_size - 1
    z_i = np.empty((n_pairs, encoded.shape[1]), dtype=np.float32)
    z_j = np.empty((n_pairs, encoded.shape[1]), dtype=np.float32)
    dt = np.empty(n_pairs, dtype=np.float32)
    for k in range(n_pairs):
        c = rng.integers(0, n_chunks)
        offset = c * chunk_size
        i = rng.integers(0, chunk_size)
        target_dt = rng.integers(1, max_dt + 1)
        j_lo = max(0, i - target_dt)
        j_hi = min(chunk_size, i + target_dt + 1)
        j_candidates = np.array([x for x in range(j_lo, j_hi) if x != i])
        if len(j_candidates) == 0:
            j = (i + 1) % chunk_size
        else:
            j = int(rng.choice(j_candidates))
        z_i[k] = encoded[offset + i]
        z_j[k] = encoded[offset + j]
        dt[k] = float(abs(i - j))
    return z_i, z_j, dt


def train_trm(
    rollout: LatentRollout,
    encoded: np.ndarray,
    chunk_size: int,
    n_pairs: int,
    n_steps: int,
    batch_size: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    smooth_l1_beta: float = 224.0,
    seed: int = 0,
) -> TRMHead:
    latent_dim = encoded.shape[1]
    head = TRMHead(latent_dim=latent_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    loss_fn = nn.SmoothL1Loss(beta=smooth_l1_beta)
    for step in range(n_steps):
        z_i_np, z_j_np, dt_np = sample_pairs(encoded, chunk_size, batch_size, rng)
        z_i = torch.from_numpy(z_i_np).to(device)
        z_j = torch.from_numpy(z_j_np).to(device)
        dt = torch.from_numpy(dt_np).to(device)
        pred = head(z_i, z_j)
        loss = loss_fn(pred, dt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == n_steps - 1:
            with torch.no_grad():
                mae = (pred - dt).abs().mean().item()
            print(
                f"  step {step:>4d}  loss={loss.item():.4f}  "
                f"mae={mae:.2f}  (target dt in [0, {chunk_size - 1}])"
            )
    return head


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a TRM reachability head on top of a frozen L0."
    )
    parser.add_argument(
        "--l0-checkpoint", type=Path, required=True,
        help="Path to the L0 checkpoint (.pt).",
    )
    parser.add_argument(
        "--shard-dir", type=Path, required=True,
        help="Directory of WebDataset .tar shards.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Where to save the trained head (state_dict .pt).",
    )
    parser.add_argument("--max-chunks", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-pairs", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Loading L0 from {args.l0_checkpoint} ...")
    rollout = LatentRollout.from_checkpoint(
        args.l0_checkpoint, device=device, gradient_policy="detach"
    )

    print(f"Encoding up to {args.max_chunks} chunks from {args.shard_dir} ...")
    encoded = encode_shards(rollout, args.shard_dir, args.max_chunks, args.chunk_size)
    print(f"Encoded shape: {encoded.shape}")

    print(f"Training TRM head for {args.n_steps} steps ...")
    head = train_trm(
        rollout=rollout,
        encoded=encoded,
        chunk_size=args.chunk_size,
        n_pairs=args.n_pairs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "latent_dim": encoded.shape[1],
            "hidden_dim": 256,
        },
        args.output,
    )
    print(f"Saved head to {args.output}")


if __name__ == "__main__":
    main()
