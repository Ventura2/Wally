"""Inspect MineRL-v0 (Zenodo v2) dataset structure."""
import json
import os
from collections import Counter

def inspect_dataset(data_dir):
    trajs = sorted(os.listdir(data_dir))
    print(f"Total trajectories: {len(trajs)}")
    
    all_action_keys = Counter()
    total_steps = 0
    
    for i, traj in enumerate(trajs[:10]):
        traj_dir = os.path.join(data_dir, traj)
        if not os.path.isdir(traj_dir):
            continue
        
        meta_path = os.path.join(traj_dir, 'metadata.json')
        npz_path = os.path.join(traj_dir, 'rendered.npz')
        mp4_path = os.path.join(traj_dir, 'recording.mp4')
        
        if not all(os.path.exists(p) for p in [meta_path, npz_path, mp4_path]):
            continue
        
        import numpy as np
        data = np.load(npz_path)
        action_keys = [k for k in data.keys() if k.startswith('action$')]
        all_action_keys.update(action_keys)
        
        steps = len(data.get('reward', []))
        total_steps += steps
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        if i < 3:
            print(f"\n--- Trajectory {i}: {traj} ---")
            print(f"  Steps: {steps}, Reward: {meta.get('total_reward')}")
            print(f"  Success: {meta.get('success')}")
            print(f"  MP4: {os.path.getsize(mp4_path) / 1024 / 1024:.1f} MB")
            print(f"  NPZ keys: {list(data.keys())}")
            for k in action_keys:
                v = data[k]
                if v.dtype.kind in ('U', 'S', 'O'):
                    from collections import Counter as C
                    vals = Counter(v.flat[:min(len(v), 100)])
                    print(f"    {k}: type={v.dtype}, sample={dict(vals.most_common(5))}")
                elif v.ndim == 1:
                    print(f"    {k}: type={v.dtype}, sum={v.sum()}/{len(v)}, range=[{v.min()}, {v.max()}]")
                else:
                    print(f"    {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")
    
    print(f"\nTotal action keys across {min(10, len(trajs))} trajectories: {dict(all_action_keys)}")
    print(f"Total steps sampled: {total_steps}")

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/workspace/data/external/navigate/MineRLNavigate-v0'
    inspect_dataset(data_dir)
