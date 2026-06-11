from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
import webdataset as wds


def find_shards(data_dir: str) -> list[str]:
    """Find all .tar shard files in a directory, including subdirectories."""
    root = Path(data_dir)
    shards = sorted(str(p) for p in root.rglob("*.tar"))
    if not shards:
        msg = f"No .tar shards found in {data_dir}"
        raise FileNotFoundError(msg)
    return shards


def decode_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Decode a WebDataset sample into frames and actions tensors.

    Handles frames/actions stored as .npy files or raw bytes.
    """
    frames: torch.Tensor | None = None
    actions: torch.Tensor | None = None

    for key, value in sample.items():
        if key.endswith(".npy"):
            arr = np.load(io.BytesIO(value) if isinstance(value, bytes) else value)
            if "frame" in key:
                frames = torch.from_numpy(arr)
            elif "action" in key:
                actions = torch.from_numpy(arr)
        elif key.endswith(".npz") or key == "npz":
            if isinstance(value, dict):
                if "frames" in value:
                    frames = torch.from_numpy(value["frames"])
                if "actions" in value:
                    actions = torch.from_numpy(value["actions"])
            else:
                data = np.load(io.BytesIO(value) if isinstance(value, bytes) else value)
                if "frames" in data:
                    frames = torch.from_numpy(data["frames"])
                if "actions" in data:
                    actions = torch.from_numpy(data["actions"])

    if frames is None or actions is None:
        msg = f"Missing frames or actions in sample keys: {list(sample.keys())}"
        raise ValueError(msg)

    if frames.dtype != torch.uint8:
        frames = frames.to(torch.uint8)
    if actions.dtype != torch.float32:
        actions = actions.to(torch.float32)

    return {"frames": frames, "actions": actions}


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """Preprocess frames: uint8->float32, normalize, resize, transpose.

    Args:
        frames: (T, H, W, 3) uint8 tensor.

    Returns:
        (T, 3, 224, 224) float32 tensor normalized to [0, 1].
    """
    x = frames.float() / 255.0
    x = x.permute(0, 3, 1, 2)  # (T, 3, H, W)

    _, _, h, w = x.shape
    if h != 224 or w != 224:
        x = F.resize(x, [224, 224])

    return x


def sample_subsequence(
    frames: torch.Tensor,
    actions: torch.Tensor,
    seq_length: int = 16,
    skip_short: bool = True,
) -> dict[str, torch.Tensor] | None:
    """Extract a random contiguous subsequence.

    Args:
        frames: (T, 3, 224, 224) preprocessed frames.
        actions: (T, A_dim) actions.
        seq_length: target subsequence length.
        skip_short: if True, skip trajectories shorter than seq_length.

    Returns:
        Dict with 'frames' and 'actions' subsequences, or None if skipped.
    """
    t = frames.shape[0]

    if t < seq_length:
        if skip_short:
            return None
        pad_len = seq_length - t
        frames = torch.cat([frames, torch.zeros(pad_len, *frames.shape[1:])], dim=0)
        actions = torch.cat([actions, torch.zeros(pad_len, *actions.shape[1:])], dim=0)
        return {"frames": frames, "actions": actions}

    start = torch.randint(0, t - seq_length + 1, (1,)).item()
    end = start + seq_length
    return {"frames": frames[start:end], "actions": actions[start:end]}


def build_pipeline(
    data_dir: str,
    seq_length: int = 16,
    skip_short: bool = True,
    shuffle: bool = True,
) -> wds.WebDataset:
    """Build a WebDataset pipeline with decoding, preprocessing, and sampling."""
    shards = find_shards(data_dir)
    dataset = wds.WebDataset(shards, shardshuffle=shuffle)

    if shuffle:
        dataset = dataset.shuffle(100)

    dataset = dataset.decode()

    def process(sample: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        decoded = decode_sample(sample)
        frames = preprocess_frames(decoded["frames"])
        actions = decoded["actions"]
        return sample_subsequence(
            frames, actions, seq_length=seq_length, skip_short=skip_short
        )

    dataset = dataset.map(process)
    dataset = dataset.select(lambda x: x is not None)

    return dataset
