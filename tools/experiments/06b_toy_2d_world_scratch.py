"""06b_toy_2d_world_scratch.py — Tier 2E (sanitized).

Re-run the toy 2-D navigation test but with a from-scratch tiny
encoder + world model, so we isolate the planner from the L0's
specific failure modes (frozen Minecraft encoder).

The architecture is intentionally similar to the L0 in spirit:
  encoder: 3-conv CNN, 64x64 -> 32 -> 16 -> 8 -> 1, then Linear
  world model: MLP (z, a) -> z'  trained on (z_t, a_t, z_{t+1})
  planner:  same CEMOptimizer, same cost fn (z_H - z_g)^2 .sum()

If the planner can't solve this trivial task, the planner (or the
training recipe) is the bug, not Minecraft.
"""
from __future__ import annotations

import sys
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

from wally.planner.cem import CEMOptimizer


# --- env -------------------------------------------------------------------
class Toy2DEnv:
    def __init__(self, size: float = 1.0) -> None:
        self.size = size
        self.pos = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.action_dim = 2

    def reset(self, pos=None, goal=None):
        if pos is None:
            pos = np.random.default_rng().uniform(-self.size, self.size, size=2).astype(np.float32)
        if goal is None:
            goal = np.random.default_rng().uniform(-self.size, self.size, size=2).astype(np.float32)
        self.pos = pos; self.goal = goal
        return self._render()

    def step(self, action):
        self.pos = np.clip(self.pos + 0.1 * action.astype(np.float32), -self.size, self.size)
        dist = float(np.linalg.norm(self.pos - self.goal))
        done = dist < 0.1
        return self._render(), -dist, done, {"dist": dist}

    def _render(self) -> np.ndarray:
        img = Image.new("RGB", (64, 64), (30, 30, 50))
        draw = ImageDraw.Draw(img)
        gx = int((self.goal[0] + self.size) / (2 * self.size) * 60 + 2)
        gy = int((self.goal[1] + self.size) / (2 * self.size) * 60 + 2)
        draw.rectangle([gx, gy, gx + 4, gy + 4], fill=(220, 30, 30))
        ax = int((self.pos[0] + self.size) / (2 * self.size) * 60 + 2)
        ay = int((self.pos[1] + self.size) / (2 * self.size) * 60 + 2)
        draw.rectangle([ax, ay, ax + 4, ay + 4], fill=(30, 220, 30))
        return np.array(img, dtype=np.uint8)


# --- tiny encoder (from scratch, mirror the L0's small size) --------------
class TinyEnc(nn.Module):
    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),       # 32
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),      # 16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),      # 8
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),      # 4
        )
        self.head = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.head(h)


# --- small world model -----------------------------------------------------
class TinyWM(nn.Module):
    def __init__(self, z_dim: int = 64, action_dim: int = 2, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


def collect(env, n, horizon, rng):
    frames = np.empty((n, horizon + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((n, horizon, 2), dtype=np.float32)
    for i in range(n):
        env.reset()
        frames[i, 0] = env._render()
        for t in range(horizon):
            a = rng.uniform(-1, 1, size=2).astype(np.float32)
            actions[i, t] = a
            f, _, _, _ = env.step(a)
            frames[i, t + 1] = f
    return frames, actions


def encode(enc, frames, device, batch=64):
    N, T, H, W, C = frames.shape
    flat = frames.reshape(N * T, H, W, C)
    x = torch.from_numpy(flat).float().permute(0, 3, 1, 2) / 255.0
    out = []
    enc.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            z = enc(x[i:i + batch].to(device))
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).reshape(N, T, -1)


def main():
    print("== Toy 2-D world (from-scratch encoder + small WM + wally CEM) ==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    env = Toy2DEnv()
    n_rollouts, horizon = 1000, 8
    print(f"collecting {n_rollouts} random rollouts ...")
    frames, actions = collect(env, n_rollouts, horizon, rng)

    enc = TinyEnc(out_dim=64).to(device)
    wm = TinyWM(z_dim=64, action_dim=2).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(wm.parameters()), lr=3e-4)

    # encode + train jointly (the encoder has to learn useful features)
    z = encode(enc, frames, device)
    print(f"  initial z shape {z.shape}, mean ||z||={np.linalg.norm(z, axis=-1).mean():.3f}")
    z_t = torch.from_numpy(z[:, :-1, :].reshape(-1, 64)).float().to(device)
    z_tp1 = torch.from_numpy(z[:, 1:, :].reshape(-1, 64)).float().to(device)
    a_t = torch.from_numpy(actions.reshape(-1, 2)).float().to(device)
    print("joint training ...")
    enc.train()
    for step in range(2000):
        idx = torch.randint(0, z_t.shape[0], (256,), device=device)
        zb, ab, zb_next = z_t[idx], a_t[idx], z_tp1[idx]
        pred = wm(zb, ab)
        loss = F.mse_loss(pred, zb_next)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"  step {step:4d}  loss={loss.item():.4f}")
    enc.eval()

    # --- planner ---------------------------------------------------------
    class Rollout:
        def __init__(self, enc, wm):
            self.enc = enc
            self.wm = wm

        def rollout(self, z_0, a):
            latents = [z_0]
            z = z_0
            for h in range(a.shape[1]):
                with torch.no_grad():
                    delta = self.wm(z, a[:, h, :]) - z
                z = (z + delta).detach()
                latents.append(z)
            return torch.stack(latents, dim=1)

        def encode(self, x):
            return self.enc(x)

    toy = Rollout(enc, wm)

    def plan_cem(z0, zg, horizon=8, pop=128, iters=8):
        cem = CEMOptimizer()
        def cost(a):
            traj = toy.rollout(z0.unsqueeze(0).expand(a.shape[0], -1), a)
            z_H = traj[:, -1, :]
            return ((z_H - zg.unsqueeze(0)) ** 2).sum(dim=-1)
        a_best, _ = cem.optimize(
            cost, horizon=horizon, action_dim=2,
            population_size=pop, n_iterations=iters,
            action_low=-1.0, action_high=1.0,
            init_mean=torch.zeros(horizon, 2, device=device),
            device=device,
        )
        return a_best

    # --- evaluate --------------------------------------------------------
    print("\nA. planner on 50 random (start, goal) pairs, horizon=8:")
    n_test = 50
    n_succ_15 = 0; n_succ_30 = 0
    mean_dist = 0.0
    mean_dist_base = 0.0
    for i in range(n_test):
        start = rng.uniform(-1, 1, size=2).astype(np.float32)
        goal = rng.uniform(-1, 1, size=2).astype(np.float32)
        mean_dist_base += float(np.linalg.norm(start - goal))
        env.pos = goal
        f_goal = torch.from_numpy(env._render()[None]).float().permute(0, 3, 1, 2).to(device)
        env.pos = start
        f_start = torch.from_numpy(env._render()[None]).float().permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            z0 = enc(f_start); zg = enc(f_goal)
        a_best = plan_cem(z0.squeeze(0), zg.squeeze(0), horizon=8)
        env.pos = start
        for h in range(8):
            env.step(a_best[h].cpu().numpy())
        d = float(np.linalg.norm(env.pos - goal))
        mean_dist += d
        if d < 0.30: n_succ_30 += 1
        if d < 0.15: n_succ_15 += 1
    print(f"  baseline (do nothing): mean dist = {mean_dist_base / n_test:.3f}")
    print(f"  planner (H=8):          mean dist = {mean_dist / n_test:.3f}")
    print(f"  planner: success @0.30 = {n_succ_30}/{n_test}  ({100*n_succ_30/n_test:.0f}%)")
    print(f"  planner: success @0.15 = {n_succ_15}/{n_test}  ({100*n_succ_15/n_test:.0f}%)")

    # With a longer horizon
    print("\nB. planner on 30 random (start, goal) pairs, horizon=16:")
    n_test = 30; mean_dist = 0.0; n_succ_15 = 0
    for i in range(n_test):
        start = rng.uniform(-1, 1, size=2).astype(np.float32)
        goal = rng.uniform(-1, 1, size=2).astype(np.float32)
        env.pos = goal
        f_goal = torch.from_numpy(env._render()[None]).float().permute(0, 3, 1, 2).to(device)
        env.pos = start
        f_start = torch.from_numpy(env._render()[None]).float().permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            z0 = enc(f_start); zg = enc(f_goal)
        a_best = plan_cem(z0.squeeze(0), zg.squeeze(0), horizon=16, pop=128, iters=10)
        env.pos = start
        for h in range(16):
            env.step(a_best[h].cpu().numpy())
        d = float(np.linalg.norm(env.pos - goal))
        mean_dist += d
        if d < 0.15: n_succ_15 += 1
    print(f"  planner (H=16):         mean dist = {mean_dist / n_test:.3f}")
    print(f"  planner: success @0.15 = {n_succ_15}/{n_test}  ({100*n_succ_15/n_test:.0f}%)")


if __name__ == "__main__":
    main()
