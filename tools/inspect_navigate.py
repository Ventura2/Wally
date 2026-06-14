"""Inspect MineRLNavigate-v0 dataset action structure."""
import json
import os
from collections import Counter
import numpy as np

data_dir = "/workspace/data/external/navigate/MineRLNavigate-v0"
trajs = sorted(os.listdir(data_dir))
print(f"Total trajectories: {len(trajs)}")

total_steps = 0
all_action_keys = Counter()

for i, traj in enumerate(trajs[:10]):
    traj_dir = os.path.join(data_dir, traj)
    if not os.path.isdir(traj_dir):
        continue
    npz_path = os.path.join(traj_dir, "rendered.npz")
    mp4_path = os.path.join(traj_dir, "recording.mp4")
    if not os.path.exists(npz_path) or not os.path.exists(mp4_path):
        continue

    data = np.load(npz_path)
    action_keys = [k for k in data.keys() if k.startswith("action$")]
    all_action_keys.update(action_keys)

    with open(os.path.join(traj_dir, "metadata.json")) as f:
        meta = json.load(f)

    steps = len(data.get("reward", []))
    total_steps += steps

    if i < 3:
        mp4_mb = os.path.getsize(mp4_path) / 1024 / 1024
        print(f"\n--- Traj {i}: {traj} ---")
        print(f"  Steps: {steps}, Reward: {meta.get('total_reward')}, MP4: {mp4_mb:.1f}MB")
        print(f"  Action keys: {action_keys}")
        for k in action_keys:
            v = data[k]
            if v.dtype.kind in ("U", "S", "O"):
                vals = Counter(v.flat[:min(len(v), 100)])
                print(f"    {k}: {dict(vals.most_common(5))}")
            elif v.ndim == 1:
                print(f"    {k}: range=[{int(v.min())}, {int(v.max())}], sum={v.sum():.0f}/{len(v)}")
            else:
                print(f"    {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

print(f"\n--- Summary ---")
print(f"Action keys across 10 trajs: {dict(all_action_keys)}")
print(f"Total steps sampled: {total_steps}")
