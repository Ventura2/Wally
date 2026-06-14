"""Transcode training shards from savez_compressed to savez (no gzip)."""

import io
import logging
import tarfile
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transcode_shard(src: Path, dst: Path) -> int:
    """Transcode a single .tar shard, return number of episodes processed."""
    count = 0
    with tarfile.open(src, "r") as src_tar, tarfile.open(dst, "w") as dst_tar:
        for member in src_tar.getmembers():
            if not member.isfile() or not member.name.endswith(".npz"):
                dst_tar.addfile(member, src_tar.extractfile(member))
                continue

            data = src_tar.extractfile(member).read()
            buf = io.BytesIO(data)
            arrays = np.load(buf)

            # Re-save without compression
            out_buf = io.BytesIO()
            np.savez(out_buf, frames=arrays["frames"], actions=arrays["actions"])
            out_data = out_buf.getvalue()

            info = tarfile.TarInfo(name=member.name)
            info.size = len(out_data)
            dst_tar.addfile(info, io.BytesIO(out_data))
            count += 1

    return count


def main() -> None:
    shard_dir = Path("/workspace/data/shards/minerl")
    tmp_dir = Path(tempfile.mkdtemp(prefix="transcode_"))
    total_eps = 0

    for shard_path in sorted(shard_dir.glob("shard_*.tar")):
        logger.info("Transcoding %s ...", shard_path.name)
        tmp_path = tmp_dir / shard_path.name
        eps = transcode_shard(shard_path, tmp_path)
        tmp_path.replace(shard_path)
        total_eps += eps
        logger.info("  -> %d episodes, size: %.1f GB", eps, shard_path.stat().st_size / 1e9)

    tmp_dir.rmdir()
    logger.info("Done. Transcoded %d episodes total.", total_eps)


if __name__ == "__main__":
    main()
