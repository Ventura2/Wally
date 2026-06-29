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
    vicreg_weight: float = 0.0
    vicreg_std_target: float = 1.0
    vicreg_cov_weight: float = 1.0
    sigreg_num_proj: int = 1024
    sigreg_knots: int = 17
    use_amp: bool = False
    checkpoint_interval: int = 1000
    log_interval: int = 10
    data_dir: str = "data/shards"
    output_dir: str = "checkpoints"
    num_workers: int = 4
    persistent_workers: bool = False
    prefetch_factor: int = 2
    skip_short: bool = True
    wandb_project: str = "wally"
    resume_from: str | None = None
    early_stop: bool = False
    early_stop_patience: int = 500
    early_stop_min_step: int = 1000
    early_stop_ema_alpha: float = 0.1
    early_stop_min_delta: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.vicreg_weight < 0:
            raise ValueError(
                f"vicreg_weight must be >= 0, got {self.vicreg_weight}"
            )
        if self.vicreg_std_target <= 0:
            raise ValueError(
                f"vicreg_std_target must be > 0, got {self.vicreg_std_target}"
            )
        if self.vicreg_cov_weight < 0:
            raise ValueError(
                f"vicreg_cov_weight must be >= 0, got {self.vicreg_cov_weight}"
            )
