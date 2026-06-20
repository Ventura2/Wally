from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agent.buffer import TrajectoryBuffer


class TestTrajectoryBufferAddAndToDict:
    def test_numpy_frames_and_actions(self) -> None:
        buf = TrajectoryBuffer()
        h, w = 64, 64
        for _ in range(5):
            frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            action = np.random.randn(25).astype(np.float32)
            buf.add(frame, action)
        result = buf.to_dict()
        assert result["frames"].shape == (5, h, w, 3)
        assert result["frames"].dtype == np.uint8
        assert result["actions"].shape == (5, 25)
        assert result["actions"].dtype == np.float32

    def test_tensor_frame_converted_to_numpy(self) -> None:
        buf = TrajectoryBuffer()
        h, w = 32, 32
        frame_tensor = torch.rand(3, h, w)
        action = np.zeros(25, dtype=np.float32)
        buf.add(frame_tensor, action)
        result = buf.to_dict()
        assert result["frames"].shape == (1, h, w, 3)
        assert result["frames"].dtype == np.uint8
        expected = (frame_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result["frames"][0], expected)

    def test_tensor_action_converted_to_numpy(self) -> None:
        buf = TrajectoryBuffer()
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        action_tensor = torch.randn(25)
        buf.add(frame, action_tensor)
        result = buf.to_dict()
        assert result["actions"].shape == (1, 25)
        np.testing.assert_allclose(result["actions"][0], action_tensor.numpy(), atol=1e-6)

    def test_event_metadata_is_recorded(self) -> None:
        buf = TrajectoryBuffer()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        action = np.zeros(25, dtype=np.float32)
        info = {
            "inventory": {"oak_log": 2, "stick": 4},
            "block_break_count": np.int64(3),
            "pov": np.zeros((8, 8, 3), dtype=np.uint8),
            "unrelated": "ignore-me",
        }
        buf.add(frame, action, info=info)

        result = buf.to_dict()
        assert "events" in result
        assert result["events"].shape == (1,)
        event = result["events"][0]
        assert event["inventory"] == {"oak_log": 2, "stick": 4}
        assert event["block_break_count"] == 3
        assert "pov" not in event
        assert "unrelated" not in event


class TestTrajectoryBufferLen:
    def test_len_empty(self) -> None:
        buf = TrajectoryBuffer()
        assert len(buf) == 0

    def test_len_after_adds(self) -> None:
        buf = TrajectoryBuffer()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        action = np.zeros(25, dtype=np.float32)
        for i in range(1, 4):
            buf.add(frame, action)
            assert len(buf) == i


class TestTrajectoryBufferReset:
    def test_reset_clears_buffer(self) -> None:
        buf = TrajectoryBuffer()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        action = np.zeros(25, dtype=np.float32)
        buf.add(frame, action)
        buf.add(frame, action)
        assert len(buf) == 2
        buf.reset()
        assert len(buf) == 0

    def test_reset_allows_reuse(self) -> None:
        buf = TrajectoryBuffer()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        action = np.zeros(25, dtype=np.float32)
        buf.add(frame, action)
        buf.reset()
        buf.add(frame, action)
        buf.add(frame, action)
        result = buf.to_dict()
        assert result["frames"].shape == (2, 8, 8, 3)
        assert result["actions"].shape == (2, 25)


class TestTrajectoryBufferEmptyToDict:
    def test_to_dict_raises_on_empty(self) -> None:
        buf = TrajectoryBuffer()
        with pytest.raises(ValueError, match="Buffer is empty"):
            buf.to_dict()
