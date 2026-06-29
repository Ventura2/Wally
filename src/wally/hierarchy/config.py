"""Configuration dataclasses for the hierarchical world-model stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LayerSpec:
    """Hyperparameters for one level (L1, L2, or L3) of the hierarchy.

    Each layer is an instance of the same :class:`JEPAWorldModel`
    architecture with different time-scale hyperparameters.

    Attributes:
        name: Human-readable layer name (``"l1"``, ``"l2"``, ``"l3"``).
        K: Time horizon in frames — how far ahead this layer predicts.
        D: Embedding dimension of the layer's state vector.
        depth: Number of transformer blocks in the predictor.
        heads: Number of attention heads in the predictor.
        drift_epsilon: Per-layer drift threshold multiplier. The actual
            threshold is ``drift_epsilon * sqrt(D)``.
    """

    name: str
    K: int
    D: int
    depth: int
    heads: int
    drift_epsilon: float

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError(f"{self.name}: K must be >= 1, got {self.K}")
        if self.D < 1:
            raise ValueError(f"{self.name}: D must be >= 1, got {self.D}")
        if self.depth < 1:
            raise ValueError(f"{self.name}: depth must be >= 1, got {self.depth}")
        if self.heads < 1:
            raise ValueError(f"{self.name}: heads must be >= 1, got {self.heads}")
        if self.drift_epsilon < 0.0:
            raise ValueError(
                f"{self.name}: drift_epsilon must be >= 0, got {self.drift_epsilon}"
            )


@dataclass
class HierarchyConfig:
    """Top-level training configuration for the hierarchy.

    Attributes:
        layers: Ordered list of layer specs. Index 0 = L1 (closest to L0).
        l0_checkpoint: Path to the frozen L0 LeWorldModel checkpoint used
            for encoder initialisation and supervision.
        lr: Optimiser learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps before the cosine schedule kicks in.
        max_steps: Total optimisation steps.
        batch_size: Mini-batch size (number of trajectory pairs per step).
        alpha: Coefficient for the SIGReg regulariser on the L1+ embedding.
        seq_length: Number of frames sampled per chunk from the shards.
        checkpoint_interval: Save a checkpoint every N steps.
        log_interval: Log metrics every N steps.
        output_dir: Directory to write checkpoints to.
        early_stop: If True, stop training when the EMA of total_loss
            stops improving (saves ``checkpoint_best.pt`` whenever the
            EMA improves; the ``L0`` trainer uses the same scheme).
        early_stop_patience: Stop after this many steps without an EMA
            improvement.
        early_stop_min_step: Don't consider stopping before this step.
        early_stop_ema_alpha: Smoothing factor for the EMA (lower = smoother).
        early_stop_min_delta: Minimum EMA improvement to count as "better".
        wandb_project: wandb project name for ``init_wandb``.
        wandb_enabled: If False, skip wandb init/logging entirely (for
            smoke tests and CI runs that don't want a wandb login).
    """

    layers: list[LayerSpec] = field(default_factory=list)
    l0_checkpoint: str = ""
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 200
    max_steps: int = 2000
    batch_size: int = 8
    alpha: float = 0.1
    seq_length: int = 128
    checkpoint_interval: int = 500
    log_interval: int = 20
    output_dir: str = "checkpoints"
    data_dir: str = "data/shards/treechop_full"
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 4
    use_concat_dataloader: bool = True
    early_stop: bool = False
    early_stop_patience: int = 500
    early_stop_min_step: int = 1000
    early_stop_ema_alpha: float = 0.1
    early_stop_min_delta: float = 0.0
    wandb_project: str = "wally"
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("HierarchyConfig.layers must contain at least one layer")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.early_stop_patience < 1:
            raise ValueError(
                f"early_stop_patience must be >= 1, got {self.early_stop_patience}"
            )
        if self.early_stop_ema_alpha <= 0.0 or self.early_stop_ema_alpha > 1.0:
            raise ValueError(
                f"early_stop_ema_alpha must be in (0, 1], got "
                f"{self.early_stop_ema_alpha}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": [vars(layer) for layer in self.layers],
            "l0_checkpoint": self.l0_checkpoint,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "alpha": self.alpha,
            "seq_length": self.seq_length,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
            "output_dir": self.output_dir,
            "data_dir": self.data_dir,
            "use_concat_dataloader": self.use_concat_dataloader,
            "early_stop": self.early_stop,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_step": self.early_stop_min_step,
            "early_stop_ema_alpha": self.early_stop_ema_alpha,
            "early_stop_min_delta": self.early_stop_min_delta,
            "wandb_project": self.wandb_project,
            "wandb_enabled": self.wandb_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchyConfig":
        layers_data = data.get("layers") or []
        layers = [LayerSpec(**layer) for layer in layers_data]
        return cls(layers=layers, **{k: v for k, v in data.items() if k != "layers"})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HierarchyConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
