"""Sample extraction and visualization for trajectory shards."""

import random
import tarfile
from pathlib import Path

from PIL import Image


def extract_samples(
    shard_path_or_dir: str | Path,
    count: int = 10,
    output_dir: str | Path = "samples_output",
) -> list[Path]:
    """Extract random observations from shards, decode JPEGs, save as PNGs.

    Args:
        shard_path_or_dir: Path to a single .tar shard or a directory
            containing .tar shards.
        count: Number of random samples to extract.
        output_dir: Directory to save PNG files.

    Returns:
        List of saved file paths.
    """
    path = Path(shard_path_or_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []
    if path.is_dir():
        shard_paths = sorted(path.glob("*.tar"))
    else:
        shard_paths = [path]

    all_entries: list[tuple[Path, str]] = []
    for shard_path in shard_paths:
        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".jpg"):
                    all_entries.append((shard_path, member.name))

    if not all_entries:
        return []

    selected = random.sample(all_entries, min(count, len(all_entries)))

    saved_paths: list[Path] = []
    for i, (shard_path, member_name) in enumerate(selected, start=1):
        with tarfile.open(shard_path, "r") as tar:
            f = tar.extractfile(member_name)
            if f is not None:
                img = Image.open(f)
                img.load()
                out_path = output_dir / f"sample_{i:03d}.png"
                img.save(out_path, "PNG")
                saved_paths.append(out_path)

    return saved_paths
