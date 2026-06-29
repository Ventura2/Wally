"""03_camera_shake_sandbox.py — Tier 1C.

For a fixed real Minecraft frame, run two experiments:

(A) Predicted-delta sensitivity: vary the camera_pitch and camera_yaw
    components of the agent action (the planner's vars at idx 0,1) and
    measure the predicted next-latent delta norm and its "camera
    direction" component (the projection onto the predicted latent
    change produced by unit camera action).

(B) Camera-shake simulation: from the same frame, run 50 random initial
    state perturbations. At each step, the L0+planner "proposes" the
    first action of a CEM plan. We measure the distribution of proposed
    camera magnitudes and movement magnitudes.

The hypothesis: if the L0 has a prior on |camera|≈1 because the
training data is full of saturated camera actions, then the predicted
delta will not strongly depend on the camera input magnitude (the
embedder saturates) and the planner's chosen camera magnitudes will
cluster near ±1.
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

from wally.planner.rollout import LatentRollout, _translate_agent_action_to_l0
from wally.planner.cem import CEMOptimizer
from wally.planner.config import CEMConfig
from wally.planner.plan import GoalConditionedPlanner

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
    print("== camera-shake sandbox ==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout = LatentRollout.from_checkpoint(CKPT, device=device)
    inner = rollout._model._model

    real_frames, _ = load_first_npz(SHARD)
    from PIL import Image
    real64 = np.stack([
        np.array(Image.fromarray(fr).resize((64, 64), Image.BILINEAR))
        for fr in real_frames[:4]
    ])
    f_t = torch.cat([to_tensor_224(r, device) for r in real64], dim=0)  # (4, 3, 224, 224)
    with torch.no_grad():
        z0 = inner.encoder(f_t)                                         # (4, 192)

    # --- A: camera-magnitude sensitivity ----------------------------------
    print("\nA. camera magnitude vs predicted delta (one-step, fixed z0):")
    magnitudes = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for axis_idx, axis_name in [(0, "pitch"), (1, "yaw")]:
        norms_per = []
        for m in magnitudes:
            a = torch.zeros(4, 25, device=device)
            a[:, axis_idx] = m
            with torch.no_grad():
                d = rollout._model.predict(z0, a)                       # (4, 192)
            norms_per.append(d.norm(dim=-1).mean().item())
        print(f"  axis={axis_name}:")
        for m, n in zip(magnitudes, norms_per):
            print(f"    mag={m:.2f}  mean ||Δz||={n:.4f}")

    # Also sweep "no movement, no attack" + camera only, then add a
    # typical movement action
    print("\nA2. with movement (forward+attack) added:")
    a_move = torch.zeros(4, 25, device=device)
    a_move[:, 2] = 1.0    # agent 2 = forward (training-vocab 0)
    a_move[:, 9] = 1.0    # agent 9 = use (training-vocab 7)
    for axis_idx, axis_name in [(0, "pitch"), (1, "yaw")]:
        for m in [0.0, 0.3, 1.0]:
            a = a_move.clone()
            a[:, axis_idx] = m
            with torch.no_grad():
                d = rollout._model.predict(z0, a)
            print(f"    axis={axis_name} mag={m:.2f}  ||Δz||={d.norm(dim=-1).mean().item():.4f}")

    # --- B: planner's first-action distribution on a fixed starting frame -
    # Build the planner the same way as the agent's planner_factory
    print("\nB. planner first-action distribution (40 CEM plans per start, 4 starts):")
    cem_config = CEMConfig.default()
    cem_config = cem_config.model_copy(
        update={
            "inventory_stall_penalty": 0.25,
            "diversity_penalty": 1.0e-3,
            "camera_still_penalty": 1.0e-3,
        }
    )
    encoder = lambda x: inner.encoder(x)
    planner = GoalConditionedPlanner(rollout, encoder, cem_config, device=device)
    # Goal = a different real frame (so the planner has something to chase)
    goal = real64[2]
    goal_t = to_tensor_224(goal, device)
    # smaller population & fewer iters to make the experiment tractable
    cem_config_fast = cem_config.model_copy(
        update={"population_size": 32, "n_iterations": 3}
    )
    planner_fast = GoalConditionedPlanner(rollout, encoder, cem_config_fast, device=device)
    # 40 plans on a fixed starting frame (use only the first of real64)
    n_plans = 40
    cam_pitch = np.empty(n_plans * 4)
    cam_yaw = np.empty(n_plans * 4)
    forward_pressed = np.empty(n_plans * 4, dtype=bool)
    attack_pressed = np.empty(n_plans * 4, dtype=bool)
    for s in range(4):
        f = real64[s]
        f_t_one = to_tensor_224(f, device)
        for k in range(n_plans):
            actions = planner_fast.plan(f_t_one, goal_t)
            # actions shape: (H, 25)
            cam_pitch[s * n_plans + k] = actions[0, 0].item()
            cam_yaw[s * n_plans + k] = actions[0, 1].item()
            forward_pressed[s * n_plans + k] = actions[0, 2].item() > 0.5
            attack_pressed[s * n_plans + k] = actions[0, 10].item() > 0.5
    print(f"  first-action camera_pitch: mean={cam_pitch.mean():+.3f}  std={cam_pitch.std():.3f}")
    print(f"    |cam_pitch|>0.5 frac: {(np.abs(cam_pitch) > 0.5).mean():.3f}")
    print(f"  first-action camera_yaw:   mean={cam_yaw.mean():+.3f}  std={cam_yaw.std():.3f}")
    print(f"    |cam_yaw|>0.5 frac:   {(np.abs(cam_yaw) > 0.5).mean():.3f}")
    print(f"  first-action forward:     pressed frac = {forward_pressed.mean():.3f}")
    print(f"  first-action attack:      pressed frac = {attack_pressed.mean():.3f}")
    # net motion vs total motion (as in user's failure report)
    print(f"  net/total for camera_pitch over all plans: "
          f"net={float(np.abs(cam_pitch.mean())):.3f}  total={float(np.abs(cam_pitch).sum()/len(cam_pitch)):.3f}")

    # plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(cam_pitch, bins=21, color="tab:purple", alpha=0.7)
    axes[0].set_title("planner first-action camera_pitch (800 plans)")
    axes[0].set_xlabel("camera_pitch in [-1, 1] (agent vocab)")
    axes[1].hist(cam_yaw, bins=21, color="tab:olive", alpha=0.7)
    axes[1].set_title("planner first-action camera_yaw (800 plans)")
    axes[1].set_xlabel("camera_yaw in [-1, 1] (agent vocab)")
    axes[2].hist2d(cam_pitch, cam_yaw, bins=21, cmap="viridis")
    axes[2].set_title("camera_pitch vs camera_yaw (plan 0)")
    axes[2].set_xlabel("camera_pitch"); axes[2].set_ylabel("camera_yaw")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_camera_shake.png", dpi=120)
    print(f"\nwrote {FIG_DIR / '03_camera_shake.png'}")

    # --- C: action_embedder saturation check (the root cause hypothesis) ---
    # If the L0 trained on clamped |cam|<=1 values, the action_embedder
    # should saturate at large |cam|.
    print("\nC. action_embedder saturation test:")
    a_test = torch.zeros(1, 1, 25, device=device)
    for m in [0.0, 0.1, 0.3, 0.5, 1.0, 5.0, 30.0, 180.0]:
        a_test[..., 10] = m   # L0 dim 10 = camera_pitch (training-vocab)
        with torch.no_grad():
            e = inner.action_embedder(a_test)
        print(f"  L0 cam_pitch={m:6.1f}  ||e||={e.norm().item():.4f}  max|e|={e.abs().max().item():.4f}")


if __name__ == "__main__":
    main()
