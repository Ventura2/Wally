"""Inspect IronPickaxe and Treechop action spaces from ZIP."""
import json
import os
import zipfile
import tempfile
from collections import Counter
import numpy as np


def inspect_zip(zip_path, limit=10):
    name = os.path.basename(zip_path).replace("-v0.zip", "")
    print(f"\n=== {name} ===")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find all traj subdirs
        traj_dirs = set()
        for name_in_zip in zf.namelist():
            parts = name_in_zip.strip("/").split("/")
            if len(parts) >= 2 and parts[1].startswith("v"):
                traj_dirs.add(parts[1])

        traj_dirs = sorted(traj_dirs)
        print(f"  Total trajectories: {len(traj_dirs)}")

        all_action_keys = Counter()
        total_steps = 0

        with tempfile.TemporaryDirectory() as tmp:
            for i, traj in enumerate(traj_dirs[:limit]):
                prefix = f"{name}/{traj}/"
                npz_path = None
                meta_path = None
                for fname in zf.namelist():
                    if fname.startswith(prefix) and fname.endswith("rendered.npz"):
                        npz_path = fname
                    elif fname.startswith(prefix) and fname.endswith("metadata.json"):
                        meta_path = fname
                if not npz_path or not meta_path:
                    continue

                zf.extract(npz_path, tmp)
                zf.extract(meta_path, tmp)

                data = np.load(os.path.join(tmp, npz_path))
                action_keys = [k for k in data.keys() if k.startswith("action$")]
                all_action_keys.update(action_keys)

                with open(os.path.join(tmp, meta_path)) as f:
                    meta = json.load(f)

                steps = len(data.get("reward", []))
                total_steps += steps

                if i < 5:
                    print(f"\n  Traj {i}: {traj} ({steps} steps, reward={meta.get('total_reward')})")
                    for k in action_keys:
                        v = data[k]
                        if v.dtype.kind in ("U", "S", "O"):
                            vals = Counter(v.flat[:min(len(v), 100)])
                            print(f"    {k}: {dict(vals.most_common(8))}")
                        elif v.ndim == 1:
                            print(f"    {k}: bin=[{int(v.min())},{int(v.max())}] sum={v.sum():.0f}/{len(v)}")
                        else:
                            print(f"    {k}: shape={v.shape} range=[{v.min():.4f},{v.max():.4f}]")

        print(f"\n  --- Summary ---")
        print(f"  Action keys seen: {dict(all_action_keys)}")
        print(f"  Total steps sampled: {total_steps}")


for path in [
    "/workspace/data/external/MineRLObtainIronPickaxe-v0.zip",
    "/workspace/data/external/MineRLTreechop-v0.zip",
    "/workspace/data/external/MineRLObtainDiamond-v0.zip",
]:
    if os.path.exists(path):
        inspect_zip(path, limit=8)
