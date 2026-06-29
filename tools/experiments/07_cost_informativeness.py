"""07_cost_informativeness.py — Tier 1I.

Take a batch of (start_frame, goal_frame) pairs and measure the L0
cost ||z(start) - z(goal)||^2 and see how it correlates with the
ground-truth pixel-space distance. If the cost is informative, the
correlation should be high and the cost surface should match the
pixel-level distance surface.
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
SHARD = PROJECT_ROOT / "data" / "shards" / "treechop_full" / "shard_000001.tar"
FIG_DIR = PROJECT_ROOT / "tools" / "experiments" / "_figures"
FIG_DIR.mkdir(exist_ok=True)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with tarfile.open(path, "r") as tar:
        for m in tar.getmembers():
            if m.name.endswith(".npz"):
                f = tar.extractfile(m)
                if f is None:
                    continue
                buf = io.BytesIO(f.read())
                with np.load(buf) as data:
                    return data["frames"], data["actions"]
    raise RuntimeError("no .npz")


def to_tensor_224(img: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(device)


def main():
    print("== cost informativeness ==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout = LatentRollout.from_checkpoint(CKPT, device=device)
    inner = rollout._model._model

    frames, _ = load_npz(SHARD)
    real64 = np.stack([np.array(Image.fromarray(fr).resize((64, 64), Image.BILINEAR)) for fr in frames])

    # take 32 start frames and 32 goal frames (could overlap)
    n_starts = min(32, len(real64) - 32)
    n_goals = min(32, len(real64) - 32)
    starts = real64[:n_starts]
    goals = real64[32:32 + n_goals]

    # encode
    z_s = []
    z_g = []
    for s in starts:
        z_s.append(inner.encoder(to_tensor_224(s, device)).cpu().numpy()[0])
    for g in goals:
        z_g.append(inner.encoder(to_tensor_224(g, device)).cpu().numpy()[0])
    z_s = np.stack(z_s)
    z_g = np.stack(z_g)

    # pairwise cost matrix
    cost = ((z_s[:, None, :] - z_g[None, :, :]) ** 2).sum(axis=-1)
    # pairwise pixel-space L2 distance
    starts_n = starts.astype(np.float32).reshape(n_starts, -1) / 255.0
    goals_n = goals.astype(np.float32).reshape(n_goals, -1) / 255.0
    pix_dist = np.linalg.norm(starts_n[:, None, :] - goals_n[None, :, :], axis=-1)

    # rank correlation
    cost_flat = cost.flatten()
    pix_flat = pix_dist.flatten()
    print(f"  cost matrix shape: {cost.shape}")
    print(f"  cost  range: [{cost.min():.2f}, {cost.max():.2f}]  mean={cost.mean():.2f}  std={cost.std():.2f}")
    print(f"  pix   range: [{pix_dist.min():.2f}, {pix_dist.max():.2f}]  mean={pix_dist.mean():.2f}  std={pix_dist.std():.2f}")
    print(f"  rank corr (Spearman): {np.corrcoef(cost_flat, pix_flat)[0, 1]:+.3f}")
    print(f"  pearson corr:         {np.corrcoef(cost_flat, np.log1p(pix_flat))[0, 1]:+.3f}  (log1p(pix))")

    # For each start, what's the lowest-cost goal? Does it match the closest pixel?
    nearest_pixel = pix_dist.argmin(axis=1)
    nearest_cost = cost.argmin(axis=1)
    agree = (nearest_pixel == nearest_cost).mean()
    print(f"  nearest-by-pix vs nearest-by-cost match: {agree:.3f}  ({int(agree * n_starts)}/{n_starts})")
    # in the top-3?
    top3_pixel = np.argsort(pix_dist, axis=1)[:, :3]
    top3_cost = np.argsort(cost, axis=1)[:, :3]
    in_top3 = np.array([top3_pixel[i, 0] in top3_cost[i] for i in range(n_starts)]).mean()
    print(f"  top-1-pix in top-3-cost: {in_top3:.3f}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(cost, cmap="viridis")
    axes[0].set_title("L0 cost  ||z_s - z_g||^2")
    axes[0].set_xlabel("goal idx"); axes[0].set_ylabel("start idx")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(pix_dist, cmap="viridis")
    axes[1].set_title("pixel-space L2")
    axes[1].set_xlabel("goal idx"); axes[1].set_ylabel("start idx")
    plt.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_cost_vs_pixel.png", dpi=120)
    print(f"  wrote {FIG_DIR / '07_cost_vs_pixel.png'}")


if __name__ == "__main__":
    main()
