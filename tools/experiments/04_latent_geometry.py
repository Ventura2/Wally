"""04_latent_geometry.py — Tier 1D.

Encode ~500 real Minecraft frames and inspect the latent geometry:
  - 2-D PCA / t-SNE scatter, colored by frame brightness, top color
  - mean latent per brightness bin
  - histogram of latent norms vs frame brightness
  - "is the L0 encoding mostly edge info or scene info?"
"""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path("D:/Projects/Personal/artificial-intelligence/wally")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wally.planner.rollout import LatentRollout

CKPT = PROJECT_ROOT / "checkpoints" / "wood_1000" / "checkpoint_1000.pt"
SHARD_DIR = PROJECT_ROOT / "data" / "shards" / "treechop_full"
FIG_DIR = PROJECT_ROOT / "tools" / "experiments" / "_figures"
FIG_DIR.mkdir(exist_ok=True)


def gather_npz_frames(shard_dir: Path, max_chunks: int = 8, per_chunk: int = 64) -> np.ndarray:
    """Open the first ``max_chunks`` chunks and return their frames."""
    out = []
    shards = sorted(shard_dir.glob("shard_*.tar"))[:2]
    for shard in shards:
        with tarfile.open(shard, "r") as tar:
            npz_count = 0
            for member in tar.getmembers():
                if member.name.endswith(".npz"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    buf = io.BytesIO(f.read())
                    with np.load(buf) as data:
                        # take a random subset of per_chunk frames
                        idx = np.random.default_rng(npz_count).choice(
                            data["frames"].shape[0], size=min(per_chunk, data["frames"].shape[0]), replace=False
                        )
                        out.append(data["frames"][idx])
                    npz_count += 1
                    if npz_count >= max_chunks:
                        break
            if npz_count >= max_chunks:
                break
    return np.concatenate(out, axis=0)


def to_tensor_224(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """img: uint8 (B, H, W, 3) or (H, W, 3). Returns (B, 3, 224, 224)."""
    single = (img.ndim == 3)
    if single:
        img = img[None]
    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    if x.shape[-1] != 224:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    if single:
        x = x.squeeze(0)
    return x.to(device)


def main() -> None:
    print("== latent geometry (PCA on 500+ real frames) ==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout = LatentRollout.from_checkpoint(CKPT, device=device)
    inner = rollout._model._model

    frames = gather_npz_frames(SHARD_DIR, max_chunks=8, per_chunk=64)
    print(f"loaded {frames.shape[0]} frames of shape {frames.shape[1:]} dtype={frames.dtype}")

    # brightness = mean over (H, W, 3)
    brightness = frames.reshape(frames.shape[0], -1).mean(axis=1)
    # top color = mean RGB over the top 20% of rows (sky region) — (N, 3)
    top_strip = frames[:, : int(frames.shape[1] * 0.2), :, :].astype(np.float32) / 255.0
    top_color = top_strip.mean(axis=(1, 2))
    # bottom color = mean RGB over bottom 30% (ground region) — (N, 3)
    bot_strip = frames[:, int(frames.shape[1] * 0.7):, :, :].astype(np.float32) / 255.0
    bot_color = bot_strip.mean(axis=(1, 2))
    # sky fraction = top-row pixel that is "sky-like" (B > 0.5 and R < 0.7)
    top_row = frames[:, 5:10, :, :].mean(axis=1) / 255.0
    sky_frac = ((top_row[..., 2] > 0.5) & (top_row[..., 0] < 0.7)).mean(axis=1)

    # encode in batches of 64
    Z = []
    inner.eval()
    with torch.no_grad():
        for i in range(0, frames.shape[0], 64):
            batch = to_tensor_224(frames[i:i + 64], device)
            z = inner.encoder(batch)
            Z.append(z.cpu().numpy())
    Z = np.concatenate(Z, axis=0)
    print(f"latent matrix: {Z.shape}")

    norms = np.linalg.norm(Z, axis=1)
    print(f"||z||  mean={norms.mean():.3f}  std={norms.std():.3f}  min={norms.min():.3f}  max={norms.max():.3f}")
    print(f"brightness  mean={brightness.mean():.1f}  std={brightness.std():.1f}")
    print(f"top RGB  mean R/G/B = {top_color[:, 0].mean():.2f}/{top_color[:, 1].mean():.2f}/{top_color[:, 2].mean():.2f}")
    print(f"bot RGB  mean R/G/B = {bot_color[:, 0].mean():.2f}/{bot_color[:, 1].mean():.2f}/{bot_color[:, 2].mean():.2f}")
    print(f"sky_frac  mean={sky_frac.mean():.3f}  std={sky_frac.std():.3f}")

    # correlations
    print("\ncorrelations of ||z|| with:")
    print(f"  brightness:      {np.corrcoef(norms, brightness)[0, 1]:+.3f}")
    print(f"  top color R:     {np.corrcoef(norms, top_color[:, 0])[0, 1]:+.3f}")
    print(f"  top color B:     {np.corrcoef(norms, top_color[:, 2])[0, 1]:+.3f}")
    print(f"  sky_frac:        {np.corrcoef(norms, sky_frac)[0, 1]:+.3f}")

    # PCA
    Xc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    print(f"\nPCA explained variance (top 10): {S[:10] ** 2 / (S ** 2).sum()}")
    coords = Xc @ Vt[:2].T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=brightness, s=8, cmap="gray")
    axes[0].set_title("L0 latent PCA-2D, color = brightness")
    plt.colorbar(sc, ax=axes[0])
    sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=top_color[:, 2] - top_color[:, 0], s=8, cmap="RdBu", vmin=-0.5, vmax=0.5)
    axes[1].set_title("L0 latent PCA-2D, color = top B - top R (blue=sky)")
    plt.colorbar(sc, ax=axes[1])
    sc = axes[2].scatter(coords[:, 0], coords[:, 1], c=sky_frac, s=8, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("L0 latent PCA-2D, color = sky_frac")
    plt.colorbar(sc, ax=axes[2])
    for a in axes:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_latent_geometry.png", dpi=120)
    print(f"\nwrote {FIG_DIR / '04_latent_geometry.png'}")

    # norm vs brightness scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(brightness, norms, s=8, alpha=0.4)
    ax.set_xlabel("frame brightness (0-255)")
    ax.set_ylabel("||z||  (L0 latent L2 norm)")
    ax.set_title("L0 latent norm vs frame brightness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_norm_vs_brightness.png", dpi=120)
    print(f"wrote {FIG_DIR / '04_norm_vs_brightness.png'}")


if __name__ == "__main__":
    main()
