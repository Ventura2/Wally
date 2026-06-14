"""Debug the extraction process for a single ZIP."""
import zipfile, os, tempfile, json
import numpy as np
import cv2

path = "/workspace/data/external/MineRLTreechop-v0.zip"
zip_name = "MineRLTreechop-v0"

with zipfile.ZipFile(path) as zf:
    traj_dirs = set()
    for n in zf.namelist():
        parts = n.strip("/").split("/")
        if len(parts) >= 2 and parts[1].startswith("v"):
            traj_dirs.add(parts[1])
    traj_dirs = sorted(traj_dirs)
    print(f"Total: {len(traj_dirs)}")
    print(f"First 5: {traj_dirs[:5]}")

    with tempfile.TemporaryDirectory() as tmp:
        for traj in traj_dirs[:5]:
            prefix = f"{zip_name}/{traj}/"
            npz_rel = meta_rel = mp4_rel = None
            for n in zf.namelist():
                rel = n[len(prefix):] if n.startswith(prefix) else ""
                if rel == "rendered.npz":
                    npz_rel = n
                elif rel == "metadata.json":
                    meta_rel = n
                elif rel == "recording.mp4":
                    mp4_rel = n
            print(f"\n{traj}:")
            print(f"  npz={npz_rel}, meta={meta_rel}, mp4={mp4_rel}")

            if not all([npz_rel, meta_rel, mp4_rel]):
                print("  SKIPPING - missing files")
                continue

            zf.extract(npz_rel, tmp)
            zf.extract(mp4_rel, tmp)

            data = np.load(os.path.join(tmp, npz_rel))
            steps = len(data.get("reward", []))
            act_keys = [k for k in data.keys() if k.startswith("action$")]
            print(f"  Steps: {steps}, action keys: {act_keys}")

            mp4_path = os.path.join(tmp, mp4_rel)
            cap = cv2.VideoCapture(mp4_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  MP4 frames: {frame_count}, FPS: {fps}")
            cap.release()

            # Read first few frames
            cap = cv2.VideoCapture(mp4_path)
            for t in range(min(3, steps)):
                ret, frame = cap.read()
                if ret:
                    print(f"  Frame {t}: shape={frame.shape}, dtype={frame.dtype}")
            cap.release()
