from __future__ import annotations

import numpy as np
import pytest

from wally.agent.viewer import FrameViewer, NullViewer


class TestNullViewer:
    def test_construction(self) -> None:
        v = NullViewer()
        assert v is not None

    def test_show_does_not_raise(self) -> None:
        v = NullViewer()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        v.show(frame)
        v.show(frame, info={"step": 1, "plan_cost": 0.1})

    def test_should_quit_always_false(self) -> None:
        v = NullViewer()
        for _ in range(5):
            v.show(np.zeros((10, 10, 3), dtype=np.uint8))
            assert v.should_quit() is False

    def test_close_does_not_raise(self) -> None:
        v = NullViewer()
        v.close()
        v.close()  # idempotent

    def test_show_with_none_pov(self) -> None:
        v = NullViewer()
        v.show(np.zeros((4, 4, 3), dtype=np.uint8))


class TestFrameViewerImportLazy:
    def test_cvl2_not_imported_on_construction(self) -> None:
        v = FrameViewer()
        assert v._cv2 is None

    def test_close_clears_cv2_reference(self) -> None:
        v = FrameViewer()
        v._cv2 = object()
        v.close()
        assert v._cv2 is None


@pytest.mark.smoke
class TestNullViewerSmoke:
    def test_null_viewer_no_exceptions(self) -> None:
        v = NullViewer()
        v.show(np.zeros((4, 4, 3), dtype=np.uint8), info={"step": 0})
        assert v.should_quit() is False
        v.close()
