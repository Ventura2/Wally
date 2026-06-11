from __future__ import annotations

import numpy as np
import pytest


class TestVoxelGrid:
    def test_set_and_get_block(self):
        from deployer.frame_renderer import VoxelGrid

        grid = VoxelGrid(size=16)
        grid.set_block(1, 2, 3, 5)
        assert grid.get_block(1, 2, 3) == 5

    def test_out_of_bounds_returns_air(self):
        from deployer.frame_renderer import VoxelGrid

        grid = VoxelGrid(size=16)
        assert grid.get_block(-1, 0, 0) == 0
        assert grid.get_block(0, 100, 0) == 0
        assert grid.get_block(16, 0, 0) == 0

    def test_set_block_out_of_bounds_is_noop(self):
        from deployer.frame_renderer import VoxelGrid

        grid = VoxelGrid(size=16)
        grid.set_block(-1, 0, 0, 5)
        grid.set_block(100, 100, 100, 5)

    def test_origin_offset(self):
        from deployer.frame_renderer import VoxelGrid

        grid = VoxelGrid(size=16)
        grid.origin = (10, 20, 30)
        grid.set_block(10, 20, 30, 3)
        assert grid.get_block(10, 20, 30) == 3
        assert grid.get_block(0, 0, 0) == 0

    def test_grid_size_property(self):
        from deployer.frame_renderer import VoxelGrid

        grid = VoxelGrid(size=64)
        assert grid.size == 64

    def test_default_block_is_air(self):
        from deployer.frame_renderer import VoxelGrid

        grid = VoxelGrid(size=16)
        assert grid.get_block(0, 0, 0) == 0

    def test_update_from_chunk(self):
        from deployer.frame_renderer import VoxelGrid

        grid = VoxelGrid(size=64)
        blocks = [[[0] * 16 for _ in range(16)] for _ in range(16)]
        blocks[0][0][0] = 1
        blocks[1][2][3] = 2
        sections = [{"y": 0, "blocks": blocks}]
        grid.update_from_chunk(0, 0, sections)
        assert grid.get_block(0, 0, 0) == 1
        assert grid.get_block(3, 1, 2) == 2


class TestFrameRenderer:
    def test_render_output_shape(self):
        from deployer.frame_renderer import FrameRenderer

        renderer = FrameRenderer(resolution=(8, 8), render_distance=1)
        image = renderer.render((8.0, 8.0, 8.0), 0.0, 0.0)
        assert image.shape == (8, 8, 3)

    def test_render_output_dtype(self):
        from deployer.frame_renderer import FrameRenderer

        renderer = FrameRenderer(resolution=(8, 8), render_distance=1)
        image = renderer.render((8.0, 8.0, 8.0), 0.0, 0.0)
        assert image.dtype == np.uint8

    def test_render_empty_grid_returns_sky(self):
        from deployer.frame_renderer import BLOCK_COLORS, FrameRenderer

        renderer = FrameRenderer(resolution=(4, 4), render_distance=1)
        image = renderer.render((8.0, 8.0, 8.0), 0.0, 0.0)
        sky = np.array(BLOCK_COLORS[0], dtype=np.uint8)
        for py in range(4):
            for px in range(4):
                np.testing.assert_array_equal(image[py, px], sky)

    def test_render_with_block_returns_color(self):
        from deployer.frame_renderer import BLOCK_COLORS, FrameRenderer

        renderer = FrameRenderer(resolution=(8, 8), render_distance=1)
        renderer.grid.set_block(8, 8, 6, 1)
        image = renderer.render((8.0, 8.0, 8.0), 0.0, 0.0)
        stone = np.array(BLOCK_COLORS[1], dtype=np.uint8)
        has_stone = any(
            np.array_equal(image[py, px], stone)
            for py in range(8)
            for px in range(8)
        )
        assert has_stone

    def test_different_resolutions(self):
        from deployer.frame_renderer import FrameRenderer

        for res in [(16, 16), (32, 24), (1, 1)]:
            renderer = FrameRenderer(resolution=res, render_distance=1)
            image = renderer.render((8.0, 8.0, 8.0), 0.0, 0.0)
            assert image.shape == (res[0], res[1], 3)

    def test_grid_property(self):
        from deployer.frame_renderer import FrameRenderer, VoxelGrid

        renderer = FrameRenderer()
        assert isinstance(renderer.grid, VoxelGrid)

    def test_update_chunk(self):
        from deployer.frame_renderer import FrameRenderer

        renderer = FrameRenderer(render_distance=2)
        blocks = [[[0] * 16 for _ in range(16)] for _ in range(16)]
        blocks[0][0][0] = 4
        sections = [{"y": 0, "blocks": blocks}]
        renderer.update_chunk(0, 0, sections)
        assert renderer.grid.get_block(0, 0, 0) == 4


class TestFramePreprocessing:
    def test_preprocess_output_shape(self):
        from deployer.frame_renderer import FrameRenderer

        renderer = FrameRenderer(resolution=(224, 224))
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        tensor = renderer.preprocess(frame)
        assert tensor.shape == (3, 224, 224)

    def test_preprocess_output_dtype(self):
        import torch

        from deployer.frame_renderer import FrameRenderer

        renderer = FrameRenderer(resolution=(224, 224))
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        tensor = renderer.preprocess(frame)
        assert tensor.dtype == torch.float32

    def test_preprocess_output_range(self):
        from deployer.frame_renderer import FrameRenderer

        renderer = FrameRenderer(resolution=(224, 224))
        frame = np.full((224, 224, 3), 255, dtype=np.uint8)
        tensor = renderer.preprocess(frame)
        assert float(tensor.min()) == pytest.approx(1.0)
        assert float(tensor.max()) == pytest.approx(1.0)

        frame_zero = np.zeros((224, 224, 3), dtype=np.uint8)
        tensor_zero = renderer.preprocess(frame_zero)
        assert float(tensor_zero.min()) == pytest.approx(0.0)

    def test_preprocess_tensor_format(self):
        from deployer.frame_renderer import FrameRenderer

        renderer = FrameRenderer(resolution=(64, 64))
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :, 0] = 128
        tensor = renderer.preprocess(frame)
        assert tensor.shape[0] == 3
        assert tensor.shape[1] == 64
        assert tensor.shape[2] == 64

    def test_preprocess_resizes_frame(self):
        from deployer.frame_renderer import FrameRenderer

        renderer = FrameRenderer(resolution=(32, 48))
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        tensor = renderer.preprocess(frame)
        assert tensor.shape == (3, 32, 48)
