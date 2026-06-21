"""Extract key frames from a recorded episode npz as PNGs.

Saves frame_0000.png, frame_0050.png, frame_0100.png, frame_0150.png,
frame_LAST.png so you can see the agent's POV progression.

Usage: uv run python tools/extract_frames.py <npz> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: extract_frames.py <npz> <out_dir>", file=sys.stderr)
        return 1
    npz_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    frames = data["frames"]
    n = len(frames)
    print(f"loaded {n} frames from {npz_path}")
    indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    for i in indices:
        img = Image.fromarray(frames[i])
        # 64x64 is small; also save a 4x upscaled version for easier viewing
        big = img.resize((256, 256), Image.NEAREST)
        out_small = out_dir / f"frame_{i:04d}.png"
        out_big = out_dir / f"frame_{i:04d}_4x.png"
        img.save(out_small)
        big.save(out_big)
        print(
            f"  step {i}: shape={frames[i].shape} -> "
            f"{out_small.name}, {out_big.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
