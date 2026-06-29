"""06c_toy_2d_world_positional.py — Tier 2E (final, clean).

Use a deterministic positional encoder (the env's (x, y) is embedded
via a fixed random Fourier feature map into a 64-dim z) and a tiny
MLP world model trained on (z, a) -> z'. This guarantees the
encoder preserves position. Now we can isolate the planner from
encoder failure and test:

  (i)  Does the planner reduce distance to a random goal?
  (ii) Does the planner reduce distance faster than random actions?
  (iii) Is the CEM converging sensibly?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("D:/Projects/Personal/artificial-intelligence/wally")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wally.planner.cem import CEMOptimizer


class Toy2DEnv:
    def __init__(self, size: float = 1.0) -> None:
        self.size = size
        self.pos = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)

    def reset(self, pos=None, goal=None):
        rng = np.random.default_rng()
        if pos is None:
            pos = rng.uniform(-self.size, self.size, size=2).astype(np.float32)
        if goal is None:
            goal = rng.uniform(-self.size, self.size, size=2).astype(np.float32)
        self.pos = pos; self.goal = goal
        return self._render()

    def step(self, action):
        self.pos = np.clip(self.pos + 0.1 * action.astype(np.float32), -self.size, self.size)
        dist = float(np.linalg.norm(self.pos - self.goal))
        return self._render(), -dist, dist < 0.1, {"dist": dist}

    def _render(self) -> np.ndarray:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (64, 64), (30, 30, 50))
        d = ImageDraw.Draw(img)
        gx = int((self.goal[0] + 1) / 2 * 60 + 2)
        gy = int((self.goal[1] + 1) / 2 * 60 + 2)
        d.rectangle([gx, gy, gx + 4, gy + 4], fill=(220, 30, 30))
        ax = int((self.pos[0] + 1) / 2 * 60 + 2)
        ay = int((self.pos[1] + 1) / 2 * 60 + 2)
        d.rectangle([ax, ay, ax + 4, ay + 4], fill=(30, 220, 30))
        return np.array(img, dtype=np.uint8)


# Deterministic Fourier-feature positional encoder
class PositionalEncoder:
    def __init__(self, pos_dim: int = 2, out_dim: int = 64, scale: float = 4.0, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # random projection (out_dim/2, pos_dim)
        self.W = rng.normal(0, scale, size=(out_dim // 2, pos_dim)).astype(np.float32)
        self.b = rng.uniform(0, 2 * np.pi, size=(out_dim // 2,)).astype(np.float32)
        self.out_dim = out_dim

    def encode(self, pos: np.ndarray) -> np.ndarray:
        # pos: (..., 2) -> z: (..., out_dim)
        proj = pos @ self.W.T + self.b
        return np.concatenate([np.cos(proj), np.sin(proj)], axis=-1)

    def __call__(self, pos: np.ndarray) -> np.ndarray:
        return self.encode(pos)


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


def collect_random(env, n, horizon, rng):
    pos_traj = np.empty((n, horizon + 1, 2), dtype=np.float32)
    actions = np.empty((n, horizon, 2), dtype=np.float32)
    for i in range(n):
        env.reset()
        pos_traj[i, 0] = env.pos
        for t in range(horizon):
            a = rng.uniform(-1, 1, size=2).astype(np.float32)
            actions[i, t] = a
            env.step(a)
            pos_traj[i, t + 1] = env.pos
    return pos_traj, actions


def main():
    print("== Toy 2-D world (positional encoder + tiny WM + wally CEM) ==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    enc = PositionalEncoder(pos_dim=2, out_dim=64, scale=4.0)
    env = Toy2DEnv()
    n_rollouts, horizon = 2000, 16
    pos, actions = collect_random(env, n_rollouts, horizon, rng)
    z = enc(pos)
    print(f"  z shape {z.shape}, mean ||z||={np.linalg.norm(z, axis=-1).mean():.3f}")

    # Verify the positional encoder actually distinguishes positions
    pos_a = np.array([[0.5, 0.5]], dtype=np.float32)
    pos_b = np.array([[0.5, 0.51]], dtype=np.float32)
    pos_c = np.array([[-0.5, 0.5]], dtype=np.float32)
    za, zb, zc = enc(pos_a), enc(pos_b), enc(pos_c)
    print(f"  ||z(0.50,0.50) - z(0.50,0.51)|| = {np.linalg.norm(za - zb):.3f}  (should be small but nonzero)")
    print(f"  ||z(0.50,0.50) - z(-0.50,0.50)|| = {np.linalg.norm(za - zc):.3f}  (should be large)")

    # train WM
    wm = TinyWM(z_dim=64, action_dim=2).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-4)
    z_t = torch.from_numpy(z[:, :-1, :].reshape(-1, 64)).float().to(device)
    z_tp1 = torch.from_numpy(z[:, 1:, :].reshape(-1, 64)).float().to(device)
    a_t = torch.from_numpy(actions.reshape(-1, 2)).float().to(device)
    for step in range(3000):
        idx = torch.randint(0, z_t.shape[0], (256,), device=device)
        pred = wm(z_t[idx], a_t[idx])
        loss = F.mse_loss(pred, z_tp1[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  WM final loss = {loss.item():.5f}")

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
                    z = self.wm(z, a[:, h, :]).detach()
                latents.append(z)
            return torch.stack(latents, dim=1)

    toy = Rollout(enc, wm)

    def plan_cem(z0, zg, horizon=8, pop=128, iters=8, init_mean=None):
        cem = CEMOptimizer()
        def cost(a):
            traj = toy.rollout(z0.unsqueeze(0).expand(a.shape[0], -1), a)
            z_H = traj[:, -1, :]
            return ((z_H - zg.unsqueeze(0)) ** 2).sum(dim=-1)
        a_best, history = cem.optimize(
            cost, horizon=horizon, action_dim=2,
            population_size=pop, n_iterations=iters,
            action_low=-1.0, action_high=1.0,
            init_mean=init_mean,
            device=device,
        )
        return a_best, history

    # --- evaluate --------------------------------------------------------
    n_test = 50
    print(f"\nA. planner on {n_test} random (start, goal) pairs, horizon=8, pop=128, iters=8:")
    n_succ_15 = 0; n_succ_30 = 0; mean_dist = 0.0; mean_dist_base = 0.0
    for i in range(n_test):
        start = rng.uniform(-1, 1, size=2).astype(np.float32)
        goal = rng.uniform(-1, 1, size=2).astype(np.float32)
        mean_dist_base += float(np.linalg.norm(start - goal))
        z0 = torch.from_numpy(enc(start[None])[0]).float().to(device)
        zg = torch.from_numpy(enc(goal[None])[0]).float().to(device)
        # hint: init the mean in the direction of the goal
        # but only "honest" planner without hints
        a_best, hist = plan_cem(z0, zg, horizon=8, pop=128, iters=8)
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

    # also: random baseline
    print(f"\nB. random action baseline (50 pairs, 8 steps of uniform random):")
    n_test = 50
    n_succ_15_r = 0; n_succ_30_r = 0; mean_dist_r = 0.0
    for i in range(n_test):
        start = rng.uniform(-1, 1, size=2).astype(np.float32)
        goal = rng.uniform(-1, 1, size=2).astype(np.float32)
        env.pos = start
        for h in range(8):
            a = rng.uniform(-1, 1, size=2).astype(np.float32)
            env.step(a)
        d = float(np.linalg.norm(env.pos - goal))
        mean_dist_r += d
        if d < 0.30: n_succ_30_r += 1
        if d < 0.15: n_succ_15_r += 1
    print(f"  random (8 steps):      mean dist = {mean_dist_r / n_test:.3f}")
    print(f"  random: success @0.30 = {n_succ_30_r}/{n_test}  ({100*n_succ_30_r/n_test:.0f}%)")
    print(f"  random: success @0.15 = {n_succ_15_r}/{n_test}  ({100*n_succ_15_r/n_test:.0f}%)")

    # C. With a long horizon
    print(f"\nC. planner on 30 pairs, horizon=20:")
    n_test = 30; mean_dist = 0.0; n_succ_15 = 0
    for i in range(n_test):
        start = rng.uniform(-1, 1, size=2).astype(np.float32)
        goal = rng.uniform(-1, 1, size=2).astype(np.float32)
        z0 = torch.from_numpy(enc(start[None])[0]).float().to(device)
        zg = torch.from_numpy(enc(goal[None])[0]).float().to(device)
        a_best, _ = plan_cem(z0, zg, horizon=20, pop=128, iters=10)
        env.pos = start
        for h in range(20):
            env.step(a_best[h].cpu().numpy())
        d = float(np.linalg.norm(env.pos - goal))
        mean_dist += d
        if d < 0.15: n_succ_15 += 1
    print(f"  planner (H=20):         mean dist = {mean_dist / n_test:.3f}")
    print(f"  planner: success @0.15 = {n_succ_15}/{n_test}  ({100*n_succ_15/n_test:.0f}%)")


if __name__ == "__main__":
    main()
