"""06_toy_2d_world.py — Tier 2E.

Build a tiny 2-D navigation world with a tiny LeWorldModel and the
SAME CEM planner that wally uses, then check whether the planner can
solve the toy env. If yes, Minecraft is the hard part. If no, the
planner has a bug independent of the L0.

Toy env: 2D continuous state (x, y) in [-1, 1]^2, action (dx, dy) in
[-1, 1]^2. A frame is rendered as a 64x64x3 RGB image showing the
agent as a green square and the goal as a red square. Frames are
fed through the L0 (frozen, untrained from checkpoint) to get a
192-dim embedding; we then train a small MLP world model on the
(t -> z) pairs from a few hundred random rollouts in the env.

We then test the CEM planner on the toy env. Success = the planner
gets the agent within 0.1 of the goal in <30 env steps.
"""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

PROJECT_ROOT = Path("D:/Projects/Personal/artificial-intelligence/wally")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wally.planner.rollout import LatentRollout
from wally.planner.cem import CEMOptimizer
from wally.planner.config import CEMConfig
from wally.planner.plan import GoalConditionedPlanner

CKPT = PROJECT_ROOT / "checkpoints" / "wood_1000" / "checkpoint_1000.pt"
FIG_DIR = PROJECT_ROOT / "tools" / "experiments" / "_figures"
FIG_DIR.mkdir(exist_ok=True)


# -------- toy env ----------------------------------------------------------
class Toy2DEnv:
    """2-D continuous position, no dynamics — `step` returns the new pos."""

    def __init__(self, size: float = 1.0) -> None:
        self.size = size
        self.pos = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.action_dim = 2

    def reset(self, pos: np.ndarray | None = None, goal: np.ndarray | None = None) -> np.ndarray:
        if pos is None:
            pos = np.random.default_rng().uniform(-self.size, self.size, size=2).astype(np.float32)
        if goal is None:
            goal = np.random.default_rng().uniform(-self.size, self.size, size=2).astype(np.float32)
        self.pos = pos
        self.goal = goal
        return self._render()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        # action in [-1, 1]^2, step size 0.1
        self.pos = np.clip(self.pos + 0.1 * action.astype(np.float32), -self.size, self.size)
        dist = float(np.linalg.norm(self.pos - self.goal))
        done = dist < 0.1
        return self._render(), -dist, done, {"pos": self.pos.copy(), "goal": self.goal.copy(), "dist": dist}

    def _render(self) -> np.ndarray:
        img = Image.new("RGB", (64, 64), (30, 30, 50))
        draw = ImageDraw.Draw(img)
        # draw goal as a red square
        gx = int((self.goal[0] + self.size) / (2 * self.size) * 60 + 2)
        gy = int((self.goal[1] + self.size) / (2 * self.size) * 60 + 2)
        draw.rectangle([gx, gy, gx + 4, gy + 4], fill=(220, 30, 30))
        # draw agent as a green square
        ax = int((self.pos[0] + self.size) / (2 * self.size) * 60 + 2)
        ay = int((self.pos[1] + self.size) / (2 * self.size) * 60 + 2)
        draw.rectangle([ax, ay, ax + 4, ay + 4], fill=(30, 220, 30))
        return np.array(img, dtype=np.uint8)


def collect_random_rollouts(env: Toy2DEnv, n: int, horizon: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Run n random rollouts of length horizon, return (frames, actions)."""
    frames = np.empty((n, horizon + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((n, horizon, env.action_dim), dtype=np.float32)
    for i in range(n):
        env.reset()
        frames[i, 0] = env._render()
        for t in range(horizon):
            a = rng.uniform(-1, 1, size=env.action_dim).astype(np.float32)
            actions[i, t] = a
            f, _, _, _ = env.step(a)
            frames[i, t + 1] = f
    return frames, actions


def encode_frames(encoder: nn.Module, frames: np.ndarray, device: torch.device) -> np.ndarray:
    """Encode a (N, T, 64, 64, 3) uint8 array into (N, T, 192) latents."""
    N, T, H, W, C = frames.shape
    flat = frames.reshape(N * T, H, W, C)
    x = torch.from_numpy(flat).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    out = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], 64):
            z = encoder(x[i:i + 64].to(device))
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).reshape(N, T, -1)


class TinyWorldModel(nn.Module):
    """A 1-step world model in the 192-dim frozen-encoder latent space.

    Input:  (z_t, action_t)  -> Output: z_{t+1} (predicted next latent).
    """

    def __init__(self, z_dim: int = 192, action_dim: int = 2, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))


def main() -> None:
    print("== Toy 2-D world with frozen L0 encoder + small MLP world model + wally CEM ==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # Load the L0 encoder (frozen)
    rollout = LatentRollout.from_checkpoint(CKPT, device=device)
    inner = rollout._model._model
    encoder = inner.encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    print("L0 encoder loaded and frozen.")

    # --- collect data -------------------------------------------------------
    env = Toy2DEnv()
    n_rollouts, horizon = 200, 8
    print(f"collecting {n_rollouts} random rollouts of horizon {horizon} ...")
    frames, actions = collect_random_rollouts(env, n_rollouts, horizon, rng)
    print(f"  frames {frames.shape}  actions {actions.shape}")

    # encode all
    print("encoding ...")
    z = encode_frames(encoder, frames, device)
    print(f"  z shape {z.shape}  mean ||z|| = {np.linalg.norm(z, axis=-1).mean():.3f}")

    # train the small world model
    z_t = z[:, :-1, :].reshape(-1, 192)
    z_tp1 = z[:, 1:, :].reshape(-1, 192)
    a_flat = actions.reshape(-1, 2)
    z_t_t = torch.from_numpy(z_t).float().to(device)
    z_tp1_t = torch.from_numpy(z_tp1).float().to(device)
    a_t = torch.from_numpy(a_flat).float().to(device)
    wm = TinyWorldModel(z_dim=192, action_dim=2).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-4)
    print("training toy world model ...")
    losses = []
    for step in range(400):
        idx = torch.randint(0, z_t_t.shape[0], (256,), device=device)
        zb, ab, zb_next = z_t_t[idx], a_t[idx], z_tp1_t[idx]
        pred = wm(zb, ab)
        loss = F.mse_loss(pred, zb_next)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"  final loss = {losses[-1]:.4f}")

    # --- build a "world model adapter" that the planner can call -----------
    # Wrap the toy wm in a class that mimics LeWorldModelAdapter's
    # predict(z, a) -> delta signature.
    class ToyWMRollout:
        def __init__(self, wm: TinyWorldModel) -> None:
            self.wm = wm

        def rollout(self, z_0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            B, H, _ = actions.shape
            latents = [z_0]
            z = z_0
            for h in range(H):
                a_h = actions[:, h, :]
                with torch.no_grad():
                    delta = self.wm(z, a_h) - z
                z_next = (z + delta).detach()
                latents.append(z_next)
                z = z_next
            return torch.stack(latents, dim=1)

        def encode(self, frame: torch.Tensor) -> torch.Tensor:
            # frame: (B, 3, 224, 224)
            return encoder(frame)

    toy_rollout = ToyWMRollout(wm)
    cem_config = CEMConfig.default().model_copy(
        update={"population_size": 64, "n_iterations": 5, "horizon": 8}
    )

    # we need to use the planner's logic; build a small custom one
    def plan_cem(z0: torch.Tensor, z_g: torch.Tensor, horizon: int = 8) -> torch.Tensor:
        cem = CEMOptimizer()
        def cost(a):
            traj = toy_rollout.rollout(z0.unsqueeze(0).expand(a.shape[0], -1), a)
            z_H = traj[:, -1, :]
            return ((z_H - z_g.unsqueeze(0)) ** 2).sum(dim=-1)
        a_best, _ = cem.optimize(
            cost,
            horizon=horizon,
            action_dim=2,
            population_size=64,
            n_iterations=5,
            action_low=-1.0,
            action_high=1.0,
            init_mean=torch.zeros(horizon, 2, device=device),
            device=device,
        )
        return a_best

    # --- test: how well does the planner actually navigate? ----------------
    print("\nA. planner vs ground-truth env navigation (50 random (start, goal) pairs):")
    n_test = 50
    successes_short, successes_med, successes_long = 0, 0, 0
    total_dist_baseline = 0.0
    total_dist_planner = 0.0
    for i in range(n_test):
        start = rng.uniform(-1, 1, size=2).astype(np.float32)
        goal = rng.uniform(-1, 1, size=2).astype(np.float32)
        # ground-truth env
        env.reset(start, goal)
        # encode start, goal
        f_start = torch.from_numpy(env._render()[None]).float().permute(0, 3, 1, 2).to(device)
        f_goal = torch.from_numpy(env._render()[None]).float().permute(0, 3, 1, 2).to(device)
        # wait — env._render() uses the env's current pos/goal, so we need to render twice with the correct values
        # better: render the goal by setting env.pos = goal temporarily
        env.pos = goal
        f_goal = torch.from_numpy(env._render()[None]).float().permute(0, 3, 1, 2).to(device)
        env.pos = start
        f_start = torch.from_numpy(env._render()[None]).float().permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            z0 = encoder(f_start)
            zg = encoder(f_goal)
        a_best = plan_cem(z0.squeeze(0), zg.squeeze(0), horizon=8)
        # rollout the chosen actions in the env
        env.pos = start
        for h in range(8):
            env.step(a_best[h].cpu().numpy())
        dist = float(np.linalg.norm(env.pos - goal))
        total_dist_planner += dist
        if dist < 0.3:
            successes_long += 1
        if dist < 0.15:
            successes_med += 1
        if dist < 0.05:
            successes_short += 1
        # baseline: no actions
        env.pos = start
        total_dist_baseline += float(np.linalg.norm(start - goal))
    print(f"  baseline (no action):     mean dist to goal = {total_dist_baseline / n_test:.3f}")
    print(f"  planner (8 steps):        mean dist to goal = {total_dist_planner / n_test:.3f}")
    print(f"  planner: success @0.30    = {successes_long}/{n_test}  ({100 * successes_long / n_test:.0f}%)")
    print(f"  planner: success @0.15    = {successes_med}/{n_test}  ({100 * successes_med / n_test:.0f}%)")
    print(f"  planner: success @0.05    = {successes_short}/{n_test}  ({100 * successes_short / n_test:.0f}%)")

    # also test with 20-step horizon
    print("\nB. with horizon=20:")
    n_test2 = 30
    successes20 = 0
    total_dist = 0.0
    for i in range(n_test2):
        start = rng.uniform(-1, 1, size=2).astype(np.float32)
        goal = rng.uniform(-1, 1, size=2).astype(np.float32)
        env.pos = goal
        f_goal = torch.from_numpy(env._render()[None]).float().permute(0, 3, 1, 2).to(device)
        env.pos = start
        f_start = torch.from_numpy(env._render()[None]).float().permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            z0 = encoder(f_start); zg = encoder(f_goal)
        a_best = plan_cem(z0.squeeze(0), zg.squeeze(0), horizon=20)
        env.pos = start
        for h in range(20):
            env.step(a_best[h].cpu().numpy())
        dist = float(np.linalg.norm(env.pos - goal))
        total_dist += dist
        if dist < 0.15:
            successes20 += 1
    print(f"  mean dist to goal = {total_dist / n_test2:.3f}")
    print(f"  success @0.15 = {successes20}/{n_test2}  ({100 * successes20 / n_test2:.0f}%)")


if __name__ == "__main__":
    main()
