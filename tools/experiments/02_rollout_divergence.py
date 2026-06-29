"""02_rollout_divergence.py — Tier 1B.

Take a real Minecraft frame and roll it out 1, 2, 4, 8, 16, 32 steps
with 100 random action sequences each (uniform in [-1, 1] in the
AGENT-vocab space — that's the planner's action space). Measure how
the predicted end-latent variance and norm grow with horizon.

The hypothesis: if the L0 is unstable over long horizons, the
predicted z_H variance should explode by horizon 8, making
replan_interval=4 no better than replan_interval=8 (the planner is
planning into noise anyway).

Also measures the "out of bounds" rate: fraction of rollouts whose
end latent has norm > 3x the training mean.
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

PROJECT_ROOT = Path("D:/Projects/Personal/artificial-intelligence/wally")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wally.planner.rollout import LatentRollout

CKPT = PROJECT_ROOT / "checkpoints" / "wood_1000" / "checkpoint_1000.pt"
SHARD = PROJECT_ROOT / "data" / "shards" / "treechop_full" / "shard_000001.tar"
FIG_DIR = PROJECT_ROOT / "tools" / "experiments" / "_figures"
FIG_DIR.mkdir(exist_ok=True)


def load_first_npz(tar_path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def to_tensor_224(img64: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img64).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x.to(device)


def main() -> None:
    print("== rollout divergence ==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout = LatentRollout.from_checkpoint(CKPT, device=device)
    inner = rollout._model._model

    # --- load 4 real starting frames -------------------------------------
    real_frames, real_actions = load_first_npz(SHARD)
    from PIL import Image
    real64 = np.stack([
        np.array(Image.fromarray(fr).resize((64, 64), Image.BILINEAR))
        for fr in real_frames[:4]
    ])
    f_t = torch.cat([to_tensor_224(r, device) for r in real64], dim=0)  # (4, 3, 224, 224)
    with torch.no_grad():
        z0_batch = inner.encoder(f_t)                                 # (4, 192)
    z0_norm = z0_batch.norm(dim=-1).mean().item()
    print(f"z0 norm: mean over 4 starts = {z0_norm:.3f}")

    # --- per-start, per-horizon rollout variance -------------------------
    horizons = [1, 2, 4, 8, 16, 32]
    n_per = 100
    print(f"horizons = {horizons}, rollouts/start = {n_per}")

    # accumulate per start
    summary = {}
    for h in horizons:
        # random action sequences in agent vocab
        a = torch.empty(4 * n_per, h, 25, device=device).uniform_(-1.0, 1.0)
        # Forward and backward: 0..9, 12..24 are 0/1 in the training data; we
        # do not clip them here to keep the experiment neutral.
        z0_rep = z0_batch.unsqueeze(1).expand(-1, n_per, -1).reshape(4 * n_per, -1)
        with torch.no_grad():
            traj = rollout.rollout(z0_rep, a)                          # (4n, h+1, 192)
        zH = traj[:, -1, :]                                            # (4n, 192)
        per_start = zH.view(4, n_per, -1)
        per_start_norm = per_start.norm(dim=-1)                        # (4, n)
        per_start_var = per_start.var(dim=1).mean(dim=-1)              # (4,) avg var
        per_start_mean_norm = per_start_norm.mean(dim=1)               # (4,)
        per_start_std_norm = per_start_norm.std(dim=1)
        oob_frac = (per_start_norm > 3 * z0_norm).float().mean(dim=1)  # (4,)
        summary[h] = {
            "mean_norm": per_start_mean_norm.mean().item(),
            "std_norm": per_start_std_norm.mean().item(),
            "var": per_start_var.mean().item(),
            "oob_frac": oob_frac.mean().item(),
        }
        print(
            f"  H={h:2d}  mean ||z_H|| = {summary[h]['mean_norm']:6.2f}  "
            f"std  = {summary[h]['std_norm']:5.2f}  var = {summary[h]['var']:7.3f}  "
            f"oob frac = {summary[h]['oob_frac']:5.2f}"
        )

    # --- also measure: how does the variance SCALE per step? -----------
    # use h=32, look at traj[:, t, :] variance across rollouts
    print("\nper-step variance along an H=32 rollout:")
    a = torch.empty(4 * n_per, 32, 25, device=device).uniform_(-1.0, 1.0)
    z0_rep = z0_batch.unsqueeze(1).expand(-1, n_per, -1).reshape(4 * n_per, -1)
    with torch.no_grad():
        traj = rollout.rollout(z0_rep, a)                              # (4n, 33, 192)
    traj = traj.view(4, n_per, 33, 192)
    for t in [0, 1, 2, 4, 8, 16, 32]:
        v = traj[:, :, t, :].var(dim=1).mean().item()
        n = traj[:, :, t, :].norm(dim=-1).mean().item()
        print(f"  t={t:2d}  mean ||z|| = {n:6.2f}  var = {v:7.3f}")

    # --- plot ---------------------------------------------------------
    Hs = list(summary.keys())
    means = [summary[h]["mean_norm"] for h in Hs]
    stds = [summary[h]["std_norm"] for h in Hs]
    oobs = [summary[h]["oob_frac"] for h in Hs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.errorbar(Hs, means, yerr=stds, marker="o", capsize=3)
    a1.axhline(3 * z0_norm, color="red", linestyle="--", label="3x training mean norm")
    a1.set_xscale("log", base=2)
    a1.set_xticks(Hs); a1.set_xticklabels([str(h) for h in Hs])
    a1.set_xlabel("horizon (steps)")
    a1.set_ylabel("||z_H||  (mean over 100 rollouts)")
    a1.set_title("Rollout end-latent norm vs horizon")
    a1.grid(True, alpha=0.3)
    a1.legend()
    a2.plot(Hs, oobs, marker="o", color="tab:red")
    a2.set_xscale("log", base=2)
    a2.set_xticks(Hs); a2.set_xticklabels([str(h) for h in Hs])
    a2.set_xlabel("horizon (steps)")
    a2.set_ylabel("OOB fraction  (||z|| > 3x train mean)")
    a2.set_title("Fraction of rollouts out of bounds")
    a2.grid(True, alpha=0.3)
    a2.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_rollout_divergence.png", dpi=120)
    print(f"\nwrote {FIG_DIR / '02_rollout_divergence.png'}")


if __name__ == "__main__":
    main()
