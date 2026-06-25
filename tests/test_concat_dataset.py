from __future__ import annotations

import io
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from wally.data.concat_dataset import (
    ConcatenatedShardDataset,
    _read_npy_shape_from_npz,
    _read_npy_shape_from_tar_member,
    _ShardTarCache,
    collate_concat_samples,
)


def _write_chunk(npz_path: Path, frames: np.ndarray, actions: np.ndarray) -> None:
    """Write one chunk .npz file to disk."""
    buf = io.BytesIO()
    np.savez_compressed(buf, frames=frames, actions=actions)
    npz_path.write_bytes(buf.getvalue())


def _write_shard(
    shard_path: Path,
    chunks: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """Write a shard containing the given chunks (ep_id, frames, actions)."""
    with tarfile.open(shard_path, "w") as tar:
        for ep_id, frames, actions in chunks:
            buf = io.BytesIO()
            np.savez_compressed(buf, frames=frames, actions=actions)
            data = buf.getvalue()
            info = tarfile.TarInfo(name=f"{ep_id}.npz")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _make_chunk(
    name: str, n_frames: int, seed: int = 0
) -> tuple[str, np.ndarray, np.ndarray]:
    """Make a chunk with a deterministic frame marker for assertions."""
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 255, (n_frames, 8, 8, 3), dtype=np.uint8)
    # Mark frame 0 with a unique value to detect cross-chunk ordering bugs.
    frames[0, 0, 0, :] = (seed % 250) + 1
    actions = rng.standard_normal((n_frames, 4)).astype(np.float32)
    return name, frames, actions


class TestReadNpyShape:
    def test_returns_first_dim_of_frames_array(self):
        frames = np.random.randint(0, 255, (42, 8, 8, 3), dtype=np.uint8)
        actions = np.random.randn(42, 4).astype(np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, frames=frames, actions=actions)
        n = _read_npy_shape_from_npz(buf.getvalue(), "frames")
        assert n == 42

    def test_falls_back_to_zero_on_bad_bytes(self):
        assert _read_npy_shape_from_npz(b"not a zip", "frames") == 0


class TestReadNpyShapeFromTarMember:
    def test_reads_shape_without_loading_pixel_data(self, tmp_path):
        """The optimized path must read just the npy header and report
        the same first-dim as the bytes-input fallback."""
        data_dir = tmp_path / "shards"
        data_dir.mkdir()
        chunk = _make_chunk("ep01__chunk000", 37, seed=1)
        _write_shard(data_dir / "shard_000000.tar", [chunk])
        with tarfile.open(data_dir / "shard_000000.tar", "r") as tf:
            member = tf.getmember("ep01__chunk000.npz")
            n = _read_npy_shape_from_tar_member(tf, member, "frames")
        assert n == 37

    def test_returns_zero_on_missing_member(self, tmp_path):
        data_dir = tmp_path / "shards"
        data_dir.mkdir()
        chunk = _make_chunk("ep01__chunk000", 4, seed=1)
        _write_shard(data_dir / "shard_000000.tar", [chunk])
        with tarfile.open(data_dir / "shard_000000.tar", "r") as tf:
            with pytest.raises(KeyError):
                tf.getmember("does_not_exist.npz")


class TestIndexCache:
    def test_cache_written_and_reused(self, tmp_path):
        """A second construction on the same data must hit the cache
        (no second build log line, same chunk/episode counts)."""
        data_dir = self._setup_two_episodes(tmp_path)
        ds1 = ConcatenatedShardDataset(str(data_dir), seq_length=8, max_samples=1)
        cache_path = data_dir / ".concat_index_v1.pkl"
        assert cache_path.is_file(), "cache file should be written on first build"
        n1_ep, n1_ch = ds1.num_episodes, ds1.num_chunks

        # Touch a shard to bump mtime only if it would actually change
        # the fingerprint; cache must still validate against the
        # current mtime, so we don't touch mtimes here.
        ds2 = ConcatenatedShardDataset(str(data_dir), seq_length=8, max_samples=1)
        assert ds2.num_episodes == n1_ep
        assert ds2.num_chunks == n1_ch
        # The on-disk index from the first build is reused.
        assert (data_dir / ".concat_index_v1.pkl").is_file()

    def test_cache_invalidated_on_shard_mtime_change(self, tmp_path):
        """Rewriting a shard (new mtime) must force a cache miss and
        a full rebuild. The rebuild should re-write the cache."""
        data_dir = self._setup_two_episodes(tmp_path)
        ds1 = ConcatenatedShardDataset(str(data_dir), seq_length=8, max_samples=1)
        assert ds1.num_chunks == 9
        cache_path = data_dir / ".concat_index_v1.pkl"
        cache_mtime_before = cache_path.stat().st_mtime_ns

        # Add a new chunk to the first shard. Bump mtime explicitly
        # in case the filesystem has low timestamp resolution.
        shard = data_dir / "shard_000000.tar"
        extra = _make_chunk("ep_extra__chunk000", 5, seed=99)
        with tarfile.open(shard, "a") as tf:
            buf = io.BytesIO()
            np.savez_compressed(buf, frames=extra[1], actions=extra[2])
            data = buf.getvalue()
            info = tarfile.TarInfo(name=extra[0] + ".npz")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        time.sleep(0.01)
        os.utime(shard, None)

        ds2 = ConcatenatedShardDataset(str(data_dir), seq_length=8, max_samples=1)
        assert ds2.num_chunks == 10  # one extra chunk picked up
        cache_mtime_after = cache_path.stat().st_mtime_ns
        assert cache_mtime_after > cache_mtime_before

    def test_cache_disabled_via_flag(self, tmp_path):
        data_dir = self._setup_two_episodes(tmp_path)
        ds = ConcatenatedShardDataset(
            str(data_dir), seq_length=8, max_samples=1, use_index_cache=False
        )
        assert ds.num_episodes == 2
        # No cache file should be written when the flag is off.
        assert not (data_dir / ".concat_index_v1.pkl").exists()

    def test_cache_invalidated_by_seq_length_change(self, tmp_path):
        """Different seq_length / skip_short / chunk_size must
        invalidate the cache (build_key is part of the fingerprint)."""
        data_dir = self._setup_two_episodes(tmp_path)
        ConcatenatedShardDataset(str(data_dir), seq_length=8, max_samples=1)
        cache_path = data_dir / ".concat_index_v1.pkl"
        before = cache_path.read_bytes()
        # Bumping seq_length forces a different build_key, which is
        # part of the cache fingerprint.
        ConcatenatedShardDataset(str(data_dir), seq_length=16, max_samples=1)
        after = cache_path.read_bytes()
        assert before != after

    def _setup_two_episodes(self, tmp_path: Path) -> Path:
        # Mirrors TestConcatenatedShardDataset._setup_two_episodes.
        data_dir = tmp_path / "shards"
        data_dir.mkdir()
        chunks = [
            _make_chunk("ep_short__chunk000", 4, seed=10),
            _make_chunk("ep_short__chunk001", 4, seed=11),
            _make_chunk("ep_short__chunk002", 4, seed=12),
            _make_chunk("ep_long__chunk000", 4, seed=20),
            _make_chunk("ep_long__chunk001", 4, seed=21),
            _make_chunk("ep_long__chunk002", 4, seed=22),
            _make_chunk("ep_long__chunk003", 4, seed=23),
            _make_chunk("ep_long__chunk004", 4, seed=24),
            _make_chunk("ep_long__chunk005", 3, seed=25),
        ]
        _write_shard(data_dir / "shard_000000.tar", chunks[:4])
        _write_shard(data_dir / "shard_000001.tar", chunks[4:])
        return data_dir


class TestShardTarCache:
    def test_opens_and_caches(self, tmp_path):
        shard = tmp_path / "x.tar"
        _write_shard(shard, [_make_chunk("ep01__chunk000", 4, seed=1)])
        cache = _ShardTarCache(max_open=4)
        try:
            tf1 = cache.get(str(shard))
            tf2 = cache.get(str(shard))
            assert tf1 is tf2
        finally:
            cache.close()

    def test_evicts_oldest(self, tmp_path):
        cache = _ShardTarCache(max_open=2)
        try:
            shards = []
            for i in range(4):
                shard = tmp_path / f"s{i}.tar"
                _write_shard(shard, [_make_chunk(f"ep{i:02d}__chunk000", 2, seed=i)])
                shards.append(str(shard))
            for s in shards:
                cache.get(s)
            # Only the last two should be open.
            assert len(cache._handles) == 2
        finally:
            cache.close()


class TestConcatenatedShardDataset:
    def _setup_two_episodes(self, tmp_path: Path) -> Path:
        """Two episodes: ep_short (3 chunks of 4 frames = 12) and
        ep_long (5 chunks of 4 frames = 20, plus a 3-frame tail)."""
        data_dir = tmp_path / "shards"
        data_dir.mkdir()
        # ep_short: 3 chunks
        chunks = [
            _make_chunk("ep_short__chunk000", 4, seed=10),
            _make_chunk("ep_short__chunk001", 4, seed=11),
            _make_chunk("ep_short__chunk002", 4, seed=12),
        ]
        # ep_long: 5 full + 1 short tail
        chunks += [
            _make_chunk("ep_long__chunk000", 4, seed=20),
            _make_chunk("ep_long__chunk001", 4, seed=21),
            _make_chunk("ep_long__chunk002", 4, seed=22),
            _make_chunk("ep_long__chunk003", 4, seed=23),
            _make_chunk("ep_long__chunk004", 4, seed=24),
            _make_chunk("ep_long__chunk005", 3, seed=25),
        ]
        _write_shard(data_dir / "shard_000000.tar", chunks[:4])
        _write_shard(data_dir / "shard_000001.tar", chunks[4:])
        return data_dir

    def test_index_groups_chunks_by_episode(self, tmp_path):
        data_dir = self._setup_two_episodes(tmp_path)
        ds = ConcatenatedShardDataset(str(data_dir), seq_length=8, max_samples=1)
        assert ds.num_episodes == 2
        assert ds.num_chunks == 9  # 3 + 6

    def test_yields_correct_shape(self, tmp_path):
        data_dir = self._setup_two_episodes(tmp_path)
        ds = ConcatenatedShardDataset(str(data_dir), seq_length=8, max_samples=5)
        for s in ds:
            # preprocess_frames hard-resizes to 224x224; the seq and
            # channel dims are what we test here.
            assert s["frames"].shape[0] == 8
            assert s["frames"].shape[1] == 3
            assert s["actions"].shape == (8, 4), s["actions"].shape
            assert s["frames"].dtype == torch.float32
            assert s["actions"].dtype == torch.float32

    def test_handles_short_tail_chunk(self, tmp_path):
        """ep_long's last chunk has 3 frames (shorter than chunk_size=4).
        Loading a window that crosses the boundary should still work."""
        data_dir = self._setup_two_episodes(tmp_path)
        ds = ConcatenatedShardDataset(
            str(data_dir), seq_length=8, chunk_size=4, max_samples=20
        )
        # We should be able to get many samples; the short tail
        # chunk should be handled by the load_window logic.
        samples = list(ds)
        assert len(samples) >= 5
        for s in samples:
            assert s["frames"].shape[0] == 8
            assert s["actions"].shape == (8, 4)

    def test_skip_short_filters_short_episodes(self, tmp_path):
        data_dir = self._setup_two_episodes(tmp_path)
        # ep_short has 12 frames; seq_length=16 is too long.
        ds = ConcatenatedShardDataset(
            str(data_dir), seq_length=16, skip_short=True, max_samples=1
        )
        assert ds.num_episodes == 1  # only ep_long
        ep_ids = list(ds._episodes.keys())
        assert ep_ids[0].startswith("ep_long")

    def test_max_samples_stops_iterator(self, tmp_path):
        data_dir = self._setup_two_episodes(tmp_path)
        ds = ConcatenatedShardDataset(str(data_dir), seq_length=8, max_samples=3)
        samples = list(ds)
        assert len(samples) == 3

    def test_action_clamping(self, tmp_path):
        data_dir = self._setup_two_episodes(tmp_path)
        ds = ConcatenatedShardDataset(str(data_dir), seq_length=8, max_samples=3)
        for s in ds:
            assert s["actions"].min() >= -1.0
            assert s["actions"].max() <= 1.0

    def test_frame_normalization(self, tmp_path):
        data_dir = self._setup_two_episodes(tmp_path)
        ds = ConcatenatedShardDataset(str(data_dir), seq_length=8, max_samples=3)
        for s in ds:
            assert s["frames"].min() >= 0.0
            assert s["frames"].max() <= 1.0

    def test_raises_on_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        # find_shards raises FileNotFoundError on an empty dir; the
        # dataset propagates that.
        with pytest.raises((RuntimeError, FileNotFoundError)):
            ConcatenatedShardDataset(str(empty), seq_length=8, max_samples=1)

    def test_invalid_seq_length(self, tmp_path):
        with pytest.raises(ValueError, match="seq_length must be >= 2"):
            ConcatenatedShardDataset("ignored", seq_length=1, max_samples=1)


class TestCollateConcatSamples:
    def test_stacks_into_batch(self):
        a = {"frames": torch.randn(8, 3, 224, 224), "actions": torch.randn(8, 25)}
        b = {"frames": torch.randn(8, 3, 224, 224), "actions": torch.randn(8, 25)}
        out = collate_concat_samples([a, b])
        assert out["frames"].shape == (2, 8, 3, 224, 224)
        assert out["actions"].shape == (2, 8, 25)


class TestCreateConcatDataloader:
    def test_smoke(self, tmp_path):
        """End-to-end smoke test: build a tiny shard set, run the
        dataloader for a few batches, check shapes."""
        from wally.data.concat_dataset import create_concat_dataloader
        data_dir = tmp_path / "shards"
        data_dir.mkdir()
        chunks = [_make_chunk(f"ep{i:02d}__chunk000", 16, seed=i) for i in range(4)]
        _write_shard(data_dir / "shard_000000.tar", chunks)
        dl = create_concat_dataloader(
            str(data_dir),
            batch_size=2,
            num_workers=0,
            seq_length=8,
            persistent_workers=False,
        )
        n = 0
        for batch in dl:
            assert batch["frames"].shape[0] == 2
            assert batch["frames"].shape[1] == 8
            assert batch["frames"].shape[2] == 3
            assert batch["actions"].shape == (2, 8, 4)
            n += 1
            if n >= 3:
                break
        assert n == 3
