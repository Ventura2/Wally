"""Training loop for the hierarchy layers (L1, L2, L3).

The training step is the same for every layer: sample random ``(t,
t + K)`` pairs from each chunk, encode both endpoints with the
layer's encoder (frozen for the lower layer, trainable linear
projection on top), and train a :class:`JEPAWorldModel` to predict
``s_{t+K}`` from ``s_t`` conditioned on ``s_{t+K}`` as the target.

Supports the same early-stopping and wandb logging pattern as the
L0 ``wally.training.trainer.Trainer`` (see
``src/wally/AGENTS.md##-early-stopping``): an EMA of ``total_loss``
triggers a stop after ``early_stop_patience`` steps without
improvement, and ``checkpoint_best.pt`` is written whenever the EMA
hits a new low.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from wally.hierarchy.config import HierarchyConfig
from wally.hierarchy.jepa import JEPAWorldModel
from wally.hierarchy.loss import combined_hierarchy_loss
from wally.training.logging import init_wandb, log_metrics
from wally.training.scheduler import create_scheduler
from wally.training.sigreg import SIGReg

logger = logging.getLogger(__name__)


@dataclass
class HierarchyTrainerState:
    """Carries the live state of a hierarchy trainer across the per-step helper."""

    global_step: int = 0
    last_log_t: float = 0.0
    scaler: torch.amp.GradScaler | None = None


class HierarchyTrainer:
    """Trainer for the L_n JEPA world model and its linear projection.

    Args:
        config: Hierarchy config (drives optimiser, scheduler, checkpointing,
            early stop, wandb).
        encoder: The layer encoder (frozen lower encoder + trainable
            linear projection). The projection's parameters are added
            to the optimiser.
        world_model: The :class:`JEPAWorldModel` predictor.
        sigreg: SIGReg regulariser applied to the projected L_n embedding.
        dataloader: A DataLoader yielding ``{"frames": (B, T, 3, 224, 224), ...}``
            batches. The trainer only consumes the ``frames`` tensor.
        device: Device to train on.
        amp_dtype: AMP dtype. ``None`` disables AMP.
    """

    def __init__(
        self,
        config: HierarchyConfig,
        encoder: torch.nn.Module,
        world_model: JEPAWorldModel,
        sigreg: SIGReg,
        dataloader: DataLoader,
        *,
        device: str | torch.device = "cuda",
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self._config = config
        self._encoder = encoder.to(device)
        self._world_model = world_model.to(device)
        self._sigreg = sigreg.to(device)
        self._dataloader = dataloader
        self._device = torch.device(device)

        trainable_params = list(self._world_model.parameters()) + [
            p for p in self._encoder.parameters() if p.requires_grad
        ]
        self._optimizer = _create_param_optimizer(
            trainable_params, lr=config.lr, weight_decay=config.weight_decay
        )

        self._scheduler = create_scheduler(
            self._optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
        self._amp_dtype = amp_dtype
        self._use_amp = amp_dtype is not None and self._device.type == "cuda"
        self._scaler: torch.amp.GradScaler | None = None
        if self._use_amp and amp_dtype == torch.float16:
            self._scaler = torch.amp.GradScaler("cuda")

        self._state = HierarchyTrainerState(
            global_step=0,
            last_log_t=time.perf_counter(),
            scaler=self._scaler,
        )

        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._K = config.layers[0].K
        self._D = config.layers[0].D

        # Early-stop state (mirrors wally.training.trainer.Trainer)
        self._ema_total_loss: float | None = None
        self._best_ema_total_loss: float = float("inf")
        self._best_step: int = 0
        self._steps_since_best: int = 0
        self._stop_training: bool = False

    @property
    def world_model(self) -> JEPAWorldModel:
        return self._world_model

    @property
    def encoder(self) -> torch.nn.Module:
        return self._encoder

    def train(self, logger: logging.Logger | None = None) -> None:
        logger = logger or logging.getLogger(self.__class__.__name__)
        self._state.global_step = 0
        self._state.last_log_t = time.perf_counter()

        if self._config.wandb_enabled:
            run_name = (
                f"{self._config.wandb_project}-{self._K}-{self._D}"
                f"-step-{self._state.global_step}"
            )
            init_wandb(
                self._config.to_dict(),
                project_name=self._config.wandb_project,
                name=run_name,
            )
            logger.info("wandb run: %s", run_name)

        if self._config.early_stop:
            logger.info(
                "Early stop: patience=%d, min_step=%d, ema_alpha=%.2f, min_delta=%.4f",
                self._config.early_stop_patience,
                self._config.early_stop_min_step,
                self._config.early_stop_ema_alpha,
                self._config.early_stop_min_delta,
            )

        keep_going = True
        while keep_going:
            for batch in self._dataloader:
                if self._state.global_step >= self._config.max_steps:
                    keep_going = False
                    break
                fetch_t = time.perf_counter()
                metrics, gpu_s, total_s = self._training_step(batch)
                step_t = time.perf_counter()
                fetch_s = fetch_t - self._state.last_log_t
                self._state.last_log_t = step_t

                next_step = self._state.global_step + 1

                wandb_metrics = dict(metrics)
                wandb_metrics["learning_rate"] = self._optimizer.param_groups[0]["lr"]
                wandb_metrics["fetch_s"] = fetch_s
                wandb_metrics["gpu_s"] = gpu_s
                wandb_metrics["total_s"] = total_s

                if (
                    next_step
                ) % self._config.log_interval == 0 or self._state.global_step == 0:
                    logger.info(
                        "Step %d | prediction_loss=%.4f | sigreg_loss=%.4f | "
                        "total_loss=%.4f | lr=%.6f | fetch=%.2fs gpu=%.3fs total=%.2fs",
                        self._state.global_step,
                        metrics["prediction_loss"],
                        metrics["sigreg_loss"],
                        metrics["total_loss"],
                        self._optimizer.param_groups[0]["lr"],
                        fetch_s,
                        gpu_s,
                        total_s,
                    )
                    if self._config.wandb_enabled:
                        log_metrics(wandb_metrics, self._state.global_step)

                if self._config.early_stop:
                    self._update_early_stop(metrics.get("total_loss", 0.0))

                if (
                    next_step % self._config.checkpoint_interval == 0
                    or next_step == self._config.max_steps
                ):
                    self._save_checkpoint(next_step)

                self._state.global_step = next_step

                if self._stop_training:
                    keep_going = False
                    break

        if self._stop_training:
            logger.info(
                "Training stopped early at step %d "
                "(best EMA total_loss=%.4f at step %d, patience=%d, "
                "min_step=%d). Use checkpoint_best.pt for the best weights.",
                self._state.global_step,
                self._best_ema_total_loss,
                self._best_step,
                self._config.early_stop_patience,
                self._config.early_stop_min_step,
            )
        else:
            logger.info(
                "Training complete. Final checkpoint saved at step %d",
                self._state.global_step,
            )

    def _update_early_stop(self, total_loss: float) -> None:
        if self._state.global_step < self._config.early_stop_min_step:
            return
        if self._ema_total_loss is None:
            self._ema_total_loss = float(total_loss)
        else:
            alpha = self._config.early_stop_ema_alpha
            self._ema_total_loss = (
                alpha * float(total_loss) + (1.0 - alpha) * self._ema_total_loss
            )
        improvement_threshold = (
            self._best_ema_total_loss - self._config.early_stop_min_delta
        )
        if self._ema_total_loss < improvement_threshold:
            self._best_ema_total_loss = self._ema_total_loss
            self._best_step = self._state.global_step
            self._steps_since_best = 0
            self._save_checkpoint_best()
            logger.info(
                "New best EMA total_loss=%.4f at step %d (saved checkpoint_best.pt)",
                self._best_ema_total_loss,
                self._state.global_step,
            )
        else:
            self._steps_since_best += 1
        if self._steps_since_best >= self._config.early_stop_patience:
            logger.info(
                "Early-stop trigger: %d steps since best (patience=%d). "
                "Best EMA total_loss=%.4f at step %d.",
                self._steps_since_best,
                self._config.early_stop_patience,
                self._best_ema_total_loss,
                self._best_step,
            )
            self._stop_training = True

    def _training_step(
        self, batch: dict[str, Tensor]
    ) -> tuple[dict[str, float], float, float]:
        start = time.perf_counter()
        frames: Tensor = batch["frames"].to(self._device, non_blocking=True)
        B, T, C, H, W = frames.shape
        if T <= self._K:
            raise ValueError(
                f"seq_length {T} must be greater than the layer horizon K={self._K}"
            )

        with torch.no_grad():
            lower = self._encode_all(frames)
        s_t = lower[:, 0, :]
        s_future = lower[:, self._K, :]

        gpu_start = time.perf_counter()
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=self._amp_dtype)
            if self._use_amp
            else _NullCtx()
        )
        with autocast_ctx:
            predicted = self._world_model.predict(s_t, s_future)
            projected = self._world_model.state_proj(s_t)
            loss, metrics = combined_hierarchy_loss(
                predicted, s_future, projected, self._config.alpha, self._sigreg
            )

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self._world_model.parameters())
                + [p for p in self._encoder.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self._world_model.parameters())
                + [p for p in self._encoder.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self._optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        gpu_s = time.perf_counter() - gpu_start
        total_s = time.perf_counter() - start
        return metrics, gpu_s, total_s

    def _encode_all(self, frames: Tensor) -> Tensor:
        """Encode a (B, T, 3, H, W) batch into (B, T, D) L_n embeddings."""
        method = getattr(self._encoder, "encode_sequence", None)
        if method is None:
            raise AttributeError(
                f"Encoder {type(self._encoder).__name__} must implement "
                f"encode_sequence()"
            )
        return method(frames)

    def _save_checkpoint(self, step: int) -> None:
        path = self._output_dir / f"checkpoint_{step}.pt"
        payload = {
            "model_state_dict": self._world_model.state_dict(),
            "encoder_state_dict": self._encoder.state_dict(),
            "global_step": step,
            "config": self._config.to_dict(),
        }
        torch.save(payload, path)

    def _save_checkpoint_best(self) -> None:
        path = self._output_dir / "checkpoint_best.pt"
        payload = {
            "model_state_dict": self._world_model.state_dict(),
            "encoder_state_dict": self._encoder.state_dict(),
            "global_step": self._state.global_step,
            "config": self._config.to_dict(),
            "best_ema_total_loss": self._best_ema_total_loss,
        }
        torch.save(payload, path)


class _NullCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


def _create_param_optimizer(
    params: list[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW with weight-decay split on bias/LayerNorm/1D params.

    Mirrors ``wally.training.optimizer.create_optimizer``.
    """
    from torch.optim import AdamW

    decay, no_decay = [], []
    for p in params:
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or p.ndim == 0:
            no_decay.append(p)
        else:
            decay.append(p)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )
