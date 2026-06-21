from __future__ import annotations

import torch
import torch.utils.data

from wally.data.dataset import build_pipeline


def collate_samples(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate a list of samples into a batch.

    Args:
        samples: list of dicts with 'frames' (T, 3, 224, 224) and 'actions' (T, A_dim).

    Returns:
        Dict with 'frames' (B, T, 3, 224, 224) and 'actions' (B, T, A_dim).
    """
    return {
        "frames": torch.stack([s["frames"] for s in samples]),
        "actions": torch.stack([s["actions"] for s in samples]),
    }


def create_dataloader(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    seq_length: int = 16,
    skip_short: bool = True,
    pin_memory: bool | None = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for WebDataset shard files.

    Args:
        data_dir: directory containing .tar shard files.
        batch_size: batch size.
        num_workers: number of data loading workers.
        seq_length: subsequence length to extract.
        skip_short: if True, skip trajectories shorter than seq_length.
        pin_memory: whether to pin host memory for async CUDA transfers.
        persistent_workers: if True, keep workers alive between epochs.
        prefetch_factor: number of batches prefetched per worker.

    Returns:
        PyTorch DataLoader yielding batches of frames and actions.
    """
    dataset = build_pipeline(
        data_dir=data_dir,
        seq_length=seq_length,
        skip_short=skip_short,
        shuffle=True,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    # PyTorch requires prefetch_factor=None when num_workers == 0.
    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False
    return torch.utils.data.DataLoader(
        dataset.batched(batch_size, collation_fn=collate_samples),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
