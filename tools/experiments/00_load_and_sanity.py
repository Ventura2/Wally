"""00_load_and_sanity.py — load the L0, run a forward pass, print report.

Used to verify the load path matches the canonical trainer before any
probes. Writes a short summary to stdout.
"""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("D:/Projects/Personal/artificial-intelligence/wally")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wally.models.lewm import LeWorldModel
from wally.planner.rollout import LatentRollout

CKPT = PROJECT_ROOT / "checkpoints" / "wood_1000" / "checkpoint_1000.pt"
SHARD = PROJECT_ROOT / "data" / "shards" / "treechop_full" / "shard_000001.tar"


def load_first_npz(tar_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Open the first .npz inside the .tar shard."""
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".npz"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                buf = io.BytesIO(f.read())
                with np.load(buf) as data:
                    return data["frames"], data["actions"]
    raise RuntimeError(f"No .npz found in {tar_path}")


def main() -> None:
    print("== L0 sanity ==")
    print(f"checkpoint: {CKPT}")
    print(f"shard:      {SHARD}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:     {device}")

    # --- load the L0 via the canonical planner adapter -----------------------
    rollout = LatentRollout.from_checkpoint(CKPT, device=device)
    inner = rollout._model._model  # LeWorldModel instance
    print("\n-- model_config (from checkpoint) --")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(ck.get("model_config"))
    print("\n-- inferred from state_dict --")
    print("encoder.is_cnn:", inner._is_cnn)
    print("encoder.embed_dim:", inner.encoder.embed_dim)
    # projector shape
    proj_in = inner.projector.net[0].in_features
    proj_out = inner.projector.net[-1].out_features
    print("projector net[0].in_features:", proj_in, " net[-1].out_features:", proj_out)
    print("predictor hidden_dim:", inner.predictor.hidden_dim)
    print("predictor depth:", inner.predictor.transformer.depth if hasattr(inner.predictor.transformer, "depth") else "?")
    print("predictor num_frames:", inner.predictor.num_frames)
    print("pred_proj out dim:", inner.pred_proj.net[-1].out_features)

    # --- load sample frames ---------------------------------------------------
    frames, actions = load_first_npz(SHARD)
    print(f"\n-- sample npz --")
    print(f"frames shape: {frames.shape}  dtype: {frames.dtype}")
    print(f"actions shape: {actions.shape}  dtype: {actions.dtype}")
    print(f"actions per-dim min/max/mean: ")
    for j in range(actions.shape[-1]):
        a = actions[..., j]
        print(f"  dim {j:2d}: min={a.min():+.3f} max={a.max():+.3f} mean={a.mean():+.3f} std={a.std():.3f} |a|>0.1 frac={float((np.abs(a) > 0.1).mean()):.3f}")

    # --- forward pass on a small batch ---------------------------------------
    T = 8
    # frames are (chunk, H, W, 3) uint8; we need (B, T, 3, 224, 224) float
    f = torch.from_numpy(frames[:T]).float() / 255.0          # (T, H, W, 3)
    f = f.permute(0, 3, 1, 2)                                # (T, 3, H, W)
    if f.shape[-1] != 224:
        f = torch.nn.functional.interpolate(f, size=(224, 224), mode="bilinear", align_corners=False)
    f = f.unsqueeze(0)                                        # (1, T, 3, 224, 224)
    a = torch.from_numpy(actions[:T]).float().unsqueeze(0)   # (1, T, 25)
    a = a.clamp(-1.0, 1.0)
    f = f.to(device); a = a.to(device)
    with torch.no_grad():
        predicted_change, emb = inner(f, a, return_embeddings=True)
    print(f"\n-- forward pass on 1×T=8 batch --")
    print(f"frames:       {tuple(f.shape)}")
    print(f"actions:      {tuple(a.shape)}")
    print(f"predicted_change: {tuple(predicted_change.shape)}  dtype={predicted_change.dtype}")
    print(f"emb:          {tuple(emb.shape)}")
    print(f"||pred_change||:  mean={predicted_change.norm(dim=-1).mean().item():.4f}  max={predicted_change.norm(dim=-1).max().item():.4f}")
    print(f"||emb||:          mean={emb.norm(dim=-1).mean().item():.4f}  max={emb.norm(dim=-1).max().item():.4f}")
    # batch norm in eval mode uses the running stats; warn if not loaded
    print(f"encoder.bn1 running_mean[:4]: {inner.encoder.bn1.running_mean[:4].tolist()}")


if __name__ == "__main__":
    main()
