"""Build training shards from existing NPZ files and continue extracting."""
import io, tarfile, logging
from pathlib import Path
from wally.data.converter import _write_shard_from_npz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

npz_dir = Path("/workspace/data/shards/minerl_temp")
output_dir = Path("/workspace/data/shards/minerl")
output_dir.mkdir(parents=True, exist_ok=True)

npz_files = sorted(npz_dir.glob("*.npz"))
logger.info("Found %d NPZ files", len(npz_files))

episodes_per_shard = 50
shard_count = 0
for i in range(0, len(npz_files), episodes_per_shard):
    chunk = npz_files[i : i + episodes_per_shard]
    shard_count += 1
    _write_shard_from_npz(chunk, output_dir, shard_count)
    logger.info("Wrote shard %d (%d eps)", shard_count, len(chunk))

logger.info("Done: %d shards written to %s", shard_count, output_dir)
