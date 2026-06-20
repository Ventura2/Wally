from unittest.mock import MagicMock

from wally.collector.buffer import TrajectoryBuffer


class TestBufferAdd:
    def test_add_returns_false_below_max(self):
        buffer = TrajectoryBuffer(max_size=5)
        for i in range(4):
            result = buffer.add({"i": i})
            assert result is False

    def test_add_returns_true_at_max(self):
        buffer = TrajectoryBuffer(max_size=3)
        buffer.add({"i": 0})
        buffer.add({"i": 1})
        result = buffer.add({"i": 2})
        assert result is True

    def test_len_returns_correct_count(self):
        buffer = TrajectoryBuffer(max_size=100)
        assert len(buffer) == 0
        buffer.add({"a": 1})
        assert len(buffer) == 1
        buffer.add({"b": 2})
        assert len(buffer) == 2


class TestBufferFlush:
    def test_flush_calls_callback(self):
        captured = []
        callback = MagicMock(side_effect=lambda data: captured.extend(data))
        buffer = TrajectoryBuffer(max_size=100, flush_callback=callback)
        buffer.add({"a": 1})
        buffer.add({"b": 2})
        count = buffer.flush()
        callback.assert_called_once()
        assert len(captured) == 2
        assert count == 2

    def test_flush_clears_buffer(self):
        buffer = TrajectoryBuffer(max_size=100)
        buffer.add({"a": 1})
        buffer.flush()
        assert len(buffer) == 0

    def test_flush_returns_zero_when_empty(self):
        buffer = TrajectoryBuffer(max_size=100)
        count = buffer.flush()
        assert count == 0

    def test_threshold_triggers_flush(self):
        callback = MagicMock()
        buffer = TrajectoryBuffer(max_size=2, flush_callback=callback)
        buffer.add({"a": 1})
        callback.assert_not_called()
        buffer.add({"b": 2})
        callback.assert_called_once()

    def test_buffer_reusable_after_flush(self):
        callback = MagicMock()
        buffer = TrajectoryBuffer(max_size=2, flush_callback=callback)
        buffer.add({"a": 1})
        buffer.add({"b": 2})
        assert callback.call_count == 1
        buffer.add({"c": 3})
        buffer.add({"d": 4})
        assert callback.call_count == 2


class TestBufferShutdown:
    def test_shutdown_flushes_remaining(self):
        captured = []
        callback = MagicMock(side_effect=lambda data: captured.extend(data))
        buffer = TrajectoryBuffer(max_size=100, flush_callback=callback)
        buffer.add({"a": 1})
        buffer.add({"b": 2})
        buffer.shutdown()
        callback.assert_called_once()
        assert len(captured) == 2

    def test_shutdown_on_empty_buffer(self):
        callback = MagicMock()
        buffer = TrajectoryBuffer(max_size=100, flush_callback=callback)
        buffer.shutdown()
        callback.assert_not_called()

    def test_flush_without_callback(self):
        buffer = TrajectoryBuffer(max_size=100)
        buffer.add({"a": 1})
        count = buffer.flush()
        assert count == 1
