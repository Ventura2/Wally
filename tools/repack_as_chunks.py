"""Repack training shards into 16-frame chunk format (uncompressed).

Reads existing compressed .npz shards, splits each episode into
non-overlapping 16-frame chunks, writes new uncompressed .npz shards.
"""

import gc
import io
import logging
import tarfile
import time
from pathlib import Path

import numpy as np

CHUNK_SIZE = 16
TARGET_BYTES = 10 * 1_000_000_000  # 10 GB per output shard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def repack(input_dir: Path, output_dir: Path) -> int:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_shards = sorted(input_dir.glob("shard_*.tar"))
    if not in_shards:
        logger.error("No shards found in %s", input_dir)
        return 0

    total_chunks = 0
    total_eps = 0
    skipped_short = 0
    out_idx = 1
    out_tar: tarfile.TarFile | None = None
    out_size = 0
    start_time = time.time()

    for shard_path in in_shards:
        logger.info("Reading %s ...", shard_path.name)
        with tarfile.open(shard_path, "r") as tar:
            members = tar.getmembers()
            for mi, member in enumerate(members):
                if not member.name.endswith(".npz"):
                    continue

                data = tar.extractfile(member).read()
                arrays = np.load(io.BytesIO(data))
                frames = arrays["frames"]  # (T, H, W, 3) uint8
                actions = arrays["actions"]  # (T, A) float16/32
                t = frames.shape[0]
                base = member.name[: -len(".npz")]
                total_eps += 1

                n_written = 0
                for ci, start in enumerate(range(0, t, CHUNK_SIZE)):
                    end = start + CHUNK_SIZE
                    if end > t:
                        skipped_short += 1
                        continue

                    chunk_frames = frames[start:end]
                    chunk_actions = actions[start:end]

                    buf = io.BytesIO()
                    np.savez(buf, frames=chunk_frames, actions=chunk_actions)
                    raw = buf.getvalue()

                    name = f"{base}_chunk{ci:04d}.npz"
                    info = tarfile.TarInfo(name=name)
                    info.size = len(raw)

                    if out_tar is None:
                        out_path = output_dir / f"shard_{out_idx:06d}.tar"
                        out_tar = tarfile.open(out_path, "w")
                        out_size = 0
                        logger.info("  Writing %s ...", out_path.name)

                    out_tar.addfile(info, io.BytesIO(raw))
                    out_size += len(raw)
                    total_chunks += 1
                    n_written += 1

                    if out_size > TARGET_BYTES:
                        elapsed = time.time() - start_time
                        rate = total_chunks / elapsed if elapsed > 0 else 0
                        logger.info(
                            "  Closed %s (%.1f GB, %d chunks, %.0f chunks/s)",
                            out_path.name,
                            out_size / 1e9,
                            n_written,
                            rate,
                        )
                        out_tar.close()
                        out_idx += 1
                        out_tar = None
                        n_written = 0

                # free per-episode memory
                del frames, actions, arrays, data
                gc.collect()

                if mi % 20 == 0 and mi > 0:
                    elapsed = time.time() - start_time
                    rate = total_chunks / elapsed if elapsed > 0 else 0
                    logger.info(
                        "  Progress: %d episodes, %d chunks, %.0f chunks/s",
                        total_eps,
                        total_chunks,
                        rate,
                    )

        gc.collect()

    if out_tar is not None:
        out_tar.close()
        logger.info("  Closed %s (%.1f GB)", f"shard_{out_idx:06d}.tar", out_size / 1e9)

    elapsed = time.time() - start_time
    logger.info(
        "Done. %d episodes → %d chunks across %d shards in %.0f s",
        total_eps,
        total_chunks,
        out_idx,
        elapsed,
    )
    logger.info("Skipped %d incomplete trailing chunks (<%d frames).", skipped_short, CHUNK_SIZE)
    return out_idx


if __name__ == "__main__":
    import sys

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/data/shards/minerl")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/workspace/data/shards/chunks")
    repack(src, dst)
