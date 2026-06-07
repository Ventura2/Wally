from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 100_000
    batch_size: int = 8
    seq_length: int = 16
    alpha: float = 0.1
    use_amp: bool = False
    checkpoint_interval: int = 1000
    log_interval: int = 10
    data_dir: str = "data/shards"
    output_dir: str = "checkpoints"
    num_workers: int = 4
    skip_short: bool = True
    wandb_project: str = "wally"
    resume_from: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
