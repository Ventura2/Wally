from dataclasses import dataclass
from typing import Tuple

import yaml


@dataclass
class CollectorConfig:
    frame_skip: int = 4
    resize: Tuple[int, int] = (224, 224)
    action_type: str = "env"
    jpeg_quality: int = 85
    buffer_size: int = 1000
    output_dir: str = "output"
    max_steps: int = 0  # 0 = unlimited, otherwise max steps per episode


def load_config(path: str) -> CollectorConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    if "resize" in data and isinstance(data["resize"], list):
        data["resize"] = tuple(data["resize"])

    return CollectorConfig(**data)
