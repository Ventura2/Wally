"""Extract and inspect a few IronPickaxe trajectories."""
import json
import os
from collections import Counter
import numpy as np


def analyze_zip(zip_path, name):
    import zipfile
    import tempfile
    import shutil

    tmp = tempfile.mkdtemp()
    try:
        print(f"\n=== {name} ===")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find trajectory directories
            dirs = set()
            for name_in_zip in zf.namelist():
                if name_in_zip.endswith("/") and name_in_zip.count("/") >= 2:
                    parts = name_in_zip.strip("/").split("/")
                    if len(parts) >= 2:
                        dirs.add(parts[1])

            traj_names = sorted(d for d in dirs if d.startswith("v"))
            print(f"Total trajectories: {len(traj_names)}")

            all_action_keys = Counter()
            total_steps = 0

            for i, traj in enumerate(traj_names[:15]):
                traj_prefix = f"{name}/{traj}/"
                npz_path = None
                meta_path = None

                for fname in zf.namelist():
                    if fname.startswith(traj_prefix) and fname.endswith("rendered.npz"):
                        npz_path = fname
                    elif fname.startswith(traj_prefix) and fname.endswith("metadata.json"):
                        meta_path = fname

                if not npz_path or not meta_path:
                    continue

                # Extract npz to temp
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
                    print(f"\n  Traj {i}: {traj}")
                    print(f"    Steps: {steps}, Reward: {meta.get('total_reward')}")
                    print(f"    Action keys: {action_keys}")
                    for k in action_keys:
                        v = data[k]
                        if v.dtype.kind in ("U", "S", "O"):
                            vals = Counter(v.flat[:min(len(v), 100)])
                            print(f"      {k}: {dict(vals.most_common(8))}")
                        elif v.ndim == 1:
                            print(f"      {k}: bin=[{int(v.min())},{int(v.max())}] sum={v.sum():.0f}/{len(v)}")
                        else:
                            print(f"      {k}: shape={v.shape} range=[{v.min():.4f},{v.max():.4f}]")

            print(f"\n  --- Summary ---")
            print(f"  Action keys: {dict(all_action_keys)}")
            print(f"  Total steps: {total_steps}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


analyze_zip("/workspace/data/external/MineRLObtainIronPickaxe-v0.zip", "MineRLObtainIronPickaxe-v0")
