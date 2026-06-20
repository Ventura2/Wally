"""First-person frame reconstruction from server chunk data."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BLOCK_COLORS: dict[int, tuple[int, int, int]] = {
    0: (135, 206, 235),
    1: (139, 119, 101),
    2: (124, 193, 82),
    3: (134, 96, 67),
    4: (128, 128, 128),
    5: (188, 152, 98),
    7: (80, 80, 80),
    8: (63, 86, 187),
    9: (63, 86, 187),
    10: (207, 92, 15),
    11: (207, 92, 15),
    12: (228, 221, 150),
    17: (103, 82, 49),
    18: (56, 148, 56),
}

DEFAULT_COLOR = (200, 200, 200)


class VoxelGrid:
    def __init__(self, size: int = 32) -> None:
        self._size = size
        self._blocks: np.ndarray = np.zeros((size, size, size), dtype=np.int32)
        self._origin: tuple[int, int, int] = (0, 0, 0)

    @property
    def size(self) -> int:
        return self._size

    @property
    def origin(self) -> tuple[int, int, int]:
        return self._origin

    @origin.setter
    def origin(self, value: tuple[int, int, int]) -> None:
        self._origin = value

    def set_block(self, x: int, y: int, z: int, block_id: int) -> None:
        lx = x - self._origin[0]
        ly = y - self._origin[1]
        lz = z - self._origin[2]
        if 0 <= lx < self._size and 0 <= ly < self._size and 0 <= lz < self._size:
            self._blocks[lx, ly, lz] = block_id

    def get_block(self, x: int, y: int, z: int) -> int:
        lx = x - self._origin[0]
        ly = y - self._origin[1]
        lz = z - self._origin[2]
        if 0 <= lx < self._size and 0 <= ly < self._size and 0 <= lz < self._size:
            return int(self._blocks[lx, ly, lz])
        return 0

    def update_from_chunk(
        self, chunk_x: int, chunk_z: int, sections: list[dict[str, Any]]
    ) -> None:
        for section in sections:
            y_base = section.get("y", 0) * 16
            blocks = section.get("blocks")
            if blocks is None:
                continue
            for local_y in range(16):
                for local_z in range(16):
                    for local_x in range(16):
                        block_id = int(
                            blocks[local_y][local_z][local_x]
                        )
                        if block_id == 0:
                            continue
                        world_x = chunk_x * 16 + local_x
                        world_y = y_base + local_y
                        world_z = chunk_z * 16 + local_z
                        self.set_block(world_x, world_y, world_z, block_id)


class FrameRenderer:
    def __init__(
        self,
        resolution: tuple[int, int] = (224, 224),
        fov: float = 70.0,
        render_distance: int = 4,
    ) -> None:
        self._resolution = resolution
        self._fov = fov
        self._grid = VoxelGrid(size=render_distance * 16)
        self._render_distance = render_distance

    @property
    def grid(self) -> VoxelGrid:
        return self._grid

    def update_chunk(
        self, chunk_x: int, chunk_z: int, sections: list[dict[str, Any]]
    ) -> None:
        self._grid.update_from_chunk(chunk_x, chunk_z, sections)

    def render(
        self,
        player_pos: tuple[float, float, float],
        yaw: float,
        pitch: float,
    ) -> np.ndarray:
        h, w = self._resolution
        image = np.zeros((h, w, 3), dtype=np.uint8)

        fov_rad = math.radians(self._fov)
        aspect = w / h

        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)

        forward = np.array([
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad),
        ])

        right = np.array([
            math.cos(yaw_rad),
            0.0,
            -math.sin(yaw_rad),
        ])
        up = np.cross(right, forward)

        for py in range(h):
            for px in range(w):
                ndc_x = (2.0 * px / w - 1.0) * math.tan(fov_rad / 2) * aspect
                ndc_y = (1.0 - 2.0 * py / h) * math.tan(fov_rad / 2)

                direction = forward + ndc_x * right + ndc_y * up
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm

                color = self._cast_ray(player_pos, direction)
                image[py, px] = color

        return image

    def _cast_ray(
        self,
        origin: tuple[float, float, float],
        direction: np.ndarray,
        max_dist: float = 64.0,
    ) -> tuple[int, int, int]:
        pos = np.array(origin, dtype=np.float64)
        step = np.sign(direction)
        step[step == 0] = 1.0

        t_delta = np.abs(1.0 / (direction + 1e-10))
        t_max = np.where(
            direction != 0,
            (np.floor(pos) + (step > 0).astype(float) - pos) / (direction + 1e-10),
            float("inf"),
        )

        for _ in range(int(max_dist)):
            block_pos = np.floor(pos).astype(int)
            block_id = self._grid.get_block(
                block_pos[0], block_pos[1], block_pos[2]
            )
            if block_id != 0:
                return BLOCK_COLORS.get(block_id, DEFAULT_COLOR)

            min_axis = int(np.argmin(t_max))
            pos[min_axis] += step[min_axis]
            t_max[min_axis] += t_delta[min_axis]

        return BLOCK_COLORS[0]

    def preprocess(self, frame: np.ndarray) -> Any:
        import torch
        from PIL import Image

        pil_image = Image.fromarray(frame).resize(
            (self._resolution[1], self._resolution[0])
        )
        resized = np.array(pil_image)
        tensor = torch.from_numpy(resized).float() / 255.0
        return tensor.permute(2, 0, 1)
