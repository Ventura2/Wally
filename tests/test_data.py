from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from wally.data.dataset import (
    find_shards,
    decode_sample,
    preprocess_frames,
    sample_subsequence,
)
from wally.data.dataloader import collate_samples


class TestFindShards:
    def test_finds_tar_files(self, tmp_path):
        (tmp_path / "shard_000000.tar").touch()
        (tmp_path / "shard_000001.tar").touch()
        (tmp_path / "other.txt").touch()
        shards = find_shards(str(tmp_path))
        assert len(shards) == 2
        assert all(s.endswith(".tar") for s in shards)

    def test_finds_nested_shards(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "shard_000000.tar").touch()
        shards = find_shards(str(tmp_path))
        assert len(shards) == 1

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .tar shards found"):
            find_shards(str(tmp_path))

    def test_sorted_output(self, tmp_path):
        (tmp_path / "b.tar").touch()
        (tmp_path / "a.tar").touch()
        (tmp_path / "c.tar").touch()
        shards = find_shards(str(tmp_path))
        assert shards == sorted(shards)


class TestDecodeSample:
    def test_decode_npy_sample(self):
        frames = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        actions = np.random.randn(4, 25).astype(np.float32)

        frames_buf = io.BytesIO()
        np.save(frames_buf, frames)
        frames_buf.seek(0)

        actions_buf = io.BytesIO()
        np.save(actions_buf, actions)
        actions_buf.seek(0)

        sample = {
            "frames.npy": frames_buf.read(),
            "actions.npy": actions_buf.read(),
        }
        result = decode_sample(sample)

        assert result["frames"].shape == (4, 64, 64, 3)
        assert result["frames"].dtype == torch.uint8
        assert result["actions"].shape == (4, 25)
        assert result["actions"].dtype == torch.float32

    def test_decode_npz_sample(self):
        frames = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        actions = np.random.randn(4, 25).astype(np.float32)

        buf = io.BytesIO()
        np.savez(buf, frames=frames, actions=actions)
        buf.seek(0)

        sample = {"data.npz": buf.read()}
        result = decode_sample(sample)

        assert result["frames"].shape == (4, 64, 64, 3)
        assert result["actions"].shape == (4, 25)

    def test_raises_on_missing_frames(self):
        sample = {"other": b"dummy"}
        with pytest.raises(ValueError, match="Missing frames or actions"):
            decode_sample(sample)


class TestPreprocessFrames:
    def test_output_shape_and_dtype(self):
        frames = torch.randint(0, 255, (8, 128, 128, 3), dtype=torch.uint8)
        out = preprocess_frames(frames)
        assert out.shape == (8, 3, 224, 224)
        assert out.dtype == torch.float32

    def test_values_normalized(self):
        frames = torch.full((4, 64, 64, 3), 255, dtype=torch.uint8)
        out = preprocess_frames(frames)
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_already_224(self):
        frames = torch.randint(0, 255, (4, 224, 224, 3), dtype=torch.uint8)
        out = preprocess_frames(frames)
        assert out.shape == (4, 3, 224, 224)


class TestSampleSubsequence:
    def test_exact_length(self):
        frames = torch.randn(16, 3, 224, 224)
        actions = torch.randn(16, 25)
        result = sample_subsequence(frames, actions, seq_length=16)
        assert result is not None
        assert result["frames"].shape == (16, 3, 224, 224)
        assert result["actions"].shape == (16, 25)

    def test_longer_trajectory(self):
        frames = torch.randn(100, 3, 224, 224)
        actions = torch.randn(100, 25)
        result = sample_subsequence(frames, actions, seq_length=16)
        assert result is not None
        assert result["frames"].shape == (16, 3, 224, 224)
        assert result["actions"].shape == (16, 25)

    def test_short_trajectory_skip(self):
        frames = torch.randn(10, 3, 224, 224)
        actions = torch.randn(10, 25)
        result = sample_subsequence(frames, actions, seq_length=16, skip_short=True)
        assert result is None

    def test_short_trajectory_pad(self):
        frames = torch.randn(10, 3, 224, 224)
        actions = torch.randn(10, 25)
        result = sample_subsequence(frames, actions, seq_length=16, skip_short=False)
        assert result is not None
        assert result["frames"].shape == (16, 3, 224, 224)
        assert result["actions"].shape == (16, 25)
        # padded portion should be zeros
        assert result["frames"][10:].sum() == 0
        assert result["actions"][10:].sum() == 0


class TestCollateSamples:
    def test_collate_shapes(self):
        samples = [
            {"frames": torch.randn(16, 3, 224, 224), "actions": torch.randn(16, 25)},
            {"frames": torch.randn(16, 3, 224, 224), "actions": torch.randn(16, 25)},
            {"frames": torch.randn(16, 3, 224, 224), "actions": torch.randn(16, 25)},
        ]
        batch = collate_samples(samples)
        assert batch["frames"].shape == (3, 16, 3, 224, 224)
        assert batch["actions"].shape == (3, 16, 25)

    def test_collate_preserves_values(self):
        frames = torch.randn(8, 3, 224, 224)
        actions = torch.randn(8, 25)
        samples = [{"frames": frames, "actions": actions}]
        batch = collate_samples(samples)
        assert torch.equal(batch["frames"][0], frames)
        assert torch.equal(batch["actions"][0], actions)
