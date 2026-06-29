"""05_action_clipping.py — Tier 1G.

Hypothesis: the L0's predicted next-latent change is more accurate
when camera actions are small. Currently the planner sends |cam| up
to 1.0 in agent-vocab, which the L0 sees as up to 180 in degrees-style
scaling.

We measure:
  (i) the predicted Δz magnitude as a function of |cam|;
  (ii) the L0's "next-frame prediction" error when the camera is
       clipped to [-0.3, 0.3] vs full [-1, 1] (in the agent-vocab,
       which is the planner's action space).

If the L0 actually learned to predict frames with small camera
deltas (because the training data was clamped to [-1, 1] and most
real camera moves are << 1 degree per frame), then the predicted
delta with a clipped camera should be smaller, and rollouts with
clipped cameras should be less divergent.
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

from wally.planner.rollout import LatentRollout, _translate_agent_action_to_l0

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


def to_tensor_224(img: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(device)


def main() -> None:
    print("== action-clipping test ==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout = LatentRollout.from_checkpoint(CKPT, device=device)
    inner = rollout._model._model

    # --- load 4 real starting frames + their true next frames ----------
    real_frames, real_actions = load_first_npz(SHARD)
    real64 = np.stack([
        np.array(Image.fromarray(fr).resize((64, 64), Image.BILINEAR))
        for fr in real_frames[:8]
    ])
    z_starts = []
    z_nexts = []
    for i in range(4):
        f0 = to_tensor_224(real64[i], device)
        f1 = to_tensor_224(real64[i + 1], device)
        with torch.no_grad():
            z_starts.append(inner.encoder(f0))
            z_nexts.append(inner.encoder(f1))
    z0 = torch.cat(z_starts, dim=0)
    z1 = torch.cat(z_nexts, dim=0)

    # --- 1-step prediction error: clipping camera dim 0,1 of the AGENT action
    print("\nA. one-step prediction error (z_pred vs z_next) as a function of camera clip:")
    a_move = torch.zeros(4, 25, device=device)
    a_move[:, 2] = 1.0     # forward (agent)
    a_move[:, 10] = 0.5    # attack (agent)
    clips = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    err_no_cam = None
    for c in clips:
        a = a_move.clone()
        if c > 0:
            a[:, 0] = c   # agent camera_pitch = +c
            a[:, 1] = c   # agent camera_yaw = +c
        with torch.no_grad():
            d = rollout._model.predict(z0, a)
            z_pred = z0 + d
            err = (z_pred - z1).pow(2).sum(dim=-1)
            mag = d.norm(dim=-1)
        print(f"  clip=±{c:.2f}  mean ||Δz||={mag.mean().item():.4f}  mean ||z_pred-z_next||^2={err.mean().item():7.2f}  per-sample: {err.tolist()}")
        if c == 0.0:
            err_no_cam = err.mean().item()

    # --- 8-step rollout: clipped vs full camera ----------------------
    print("\nB. 8-step rollout with camera clipped at ±0.3 vs ±1.0:")
    H = 8
    n_pert = 30
    rng = torch.Generator(device=device).manual_seed(0)
    # build a base random action sequence; perturb camera at each step
    a_base = torch.empty(4 * n_pert, H, 25, device=device).uniform_(-1.0, 1.0, generator=rng)
    a_base[..., :12] = (a_base[..., :12] > 0.5).float()  # binarize button dims
    a_base[..., 12:] = 0.0                              # zero inventory dims

    # also make a "no movement" version with random camera
    a_cam = torch.zeros(4 * n_pert, H, 25, device=device).uniform_(-1.0, 1.0, generator=rng)
    a_cam[..., 0] = torch.empty(4 * n_pert, H, device=device).uniform_(0.0, 1.0, generator=rng)
    a_cam[..., 1] = torch.empty(4 * n_pert, H, device=device).uniform_(0.0, 1.0, generator=rng)

    z0_rep = z0.unsqueeze(1).expand(-1, n_pert, -1).reshape(4 * n_pert, -1)

    for label, actions in [
        ("random full [-1, 1]  (binary buttons)", a_base),
        ("camera only [-1, 1]", a_cam),
        ("camera only clipped ±0.3", a_cam.clamp(-0.3, 0.3)),
        ("camera only clipped ±0.1", a_cam.clamp(-0.1, 0.1)),
    ]:
        with torch.no_grad():
            traj = rollout.rollout(z0_rep, actions)
        zH = traj[:, -1, :]
        norms = zH.norm(dim=-1)
        diverged = (norms > 3 * norms[:4].mean()).float().mean().item()
        # also: how far did the rollout move from z0
        dist = (zH - z0_rep).norm(dim=-1)
        # what is the average per-step Δz magnitude?
        per_step = (traj[:, 1:, :] - traj[:, :-1, :]).norm(dim=-1).mean()
        print(f"  {label:42s}  ||z_H|| mean={norms.mean().item():.2f}  "
              f"||z_H - z0|| mean={dist.mean().item():.2f}  "
              f"oob frac={diverged:.2f}  "
              f"per-step ||Δz||={per_step.item():.4f}")


if __name__ == "__main__":
    main()
