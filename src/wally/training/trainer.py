from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from wally.training.checkpoint import load_checkpoint, save_checkpoint
from wally.training.logging import init_wandb, log_metrics
from wally.training.losses import combined_loss
from wally.training.optimizer import create_optimizer
from wally.training.scheduler import create_scheduler
from wally.training.sigreg import SIGReg

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop orchestrator for LeWorldModel with closed-form SIGReg."""

    def __init__(
        self,
        model: nn.Module,
        sigreg: SIGReg,
        train_loader: DataLoader[Any],
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.sigreg = sigreg
        self.train_loader = train_loader
        self.config = config

        self.lr: float = config.get("lr", 1e-4)
        self.weight_decay: float = config.get("weight_decay", 1e-5)
        self.warmup_steps: int = config.get("warmup_steps", 1000)
        self.max_steps: int = config.get("max_steps", 100_000)
        self.alpha: float = config.get("alpha", 0.1)
        self.use_amp: bool = config.get("use_amp", False)
        # amp_dtype: "bfloat16" (stable, no scaler) or "float16" (needs GradScaler)
        self.amp_dtype: torch.dtype = (
            torch.bfloat16
            if config.get("amp_dtype", "bfloat16") == "bfloat16"
            else torch.float16
        )
        # GradScaler is only needed for FP16; BF16 has the same range as FP32
        self.use_scaler: bool = self.use_amp and self.amp_dtype == torch.float16
        self.checkpoint_interval: int = config.get("checkpoint_interval", 1000)
        self.log_interval: int = config.get("log_interval", 10)
        self.output_dir: Path = Path(config.get("output_dir", "checkpoints"))

        default_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device: torch.device = config.get("device", default_device)

        self.model.to(self.device)
        self.sigreg.to(self.device)

        self.optimizer: Optimizer = create_optimizer(
            self.model, self.lr, self.weight_decay
        )
        self.scheduler = create_scheduler(
            self.optimizer, self.warmup_steps, self.max_steps
        )

        self.scaler: GradScaler | None = (
            GradScaler("cuda") if self.use_scaler else None
        )
        self.global_step: int = 0

    def _training_step(
        self, frames: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, float]:
        """Execute a single training step.

        Args:
            frames:  (B, T, 3, 224, 224)
            actions: (B, T, action_dim)

        Returns:
            Metrics dict with prediction_loss, sigreg_loss, total_loss.
        """
        frames = frames.to(self.device)
        actions = actions.to(self.device)
        frames = torch.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward pass
        if self.use_amp:
            with autocast("cuda", dtype=self.amp_dtype):
                predicted, target, embeddings = self.model(
                    frames, actions, return_embeddings=True
                )
                total_loss, metrics = combined_loss(
                    predicted, target, embeddings, self.alpha, self.sigreg
                )
        else:
            predicted, target, embeddings = self.model(
                frames, actions, return_embeddings=True
            )
            total_loss, metrics = combined_loss(
                predicted, target, embeddings, self.alpha, self.sigreg
            )

        if not torch.isfinite(total_loss):
            logger.warning(
                "Step %d: non-finite loss %.4f, skipping update",
                self.global_step,
                total_loss.item(),
            )
            self.optimizer.zero_grad()
            with warnings.catch_warnings():
                # Suppress "lr_scheduler.step() before optimizer.step()"
                # warning: this path intentionally skips opt.step.
                warnings.simplefilter("ignore", UserWarning)
                self.scheduler.step()
            self.global_step += 1
            metrics.setdefault("prediction_loss", float("nan"))
            metrics.setdefault("sigreg_loss", float("nan"))
            metrics.setdefault("total_loss", float("nan"))
            metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
            return metrics

        # Backward pass for main model
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # Guard against non-finite gradients: an inf/nan grad poisons the
            # optimizer state (exp_avg, exp_avg_sq) and turns the *next* forward
            # into NaN. Detect and skip the step before the damage is done.
            # NOTE: do NOT call scaler.step() / optimizer.step() in the skip
            # path — scaler.step() has its own inf check on SCALED gradients
            # that may pass even when unscaled gradients are non-finite, which
            # would apply a bad update and poison the model.
            grads_finite = all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in self.model.parameters()
                if p.grad is not None
            )
            if not grads_finite:
                nan_param_count = sum(
                    1 for p in self.model.parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                )
                logger.warning(
                    "Step %d: %d params with non-finite grad, skipping update",
                    self.global_step,
                    nan_param_count,
                )
                self.optimizer.zero_grad()
                with warnings.catch_warnings():
                    # Suppress "lr_scheduler.step() before optimizer.step()"
                    # warning: this path intentionally skips opt.step.
                    warnings.simplefilter("ignore", UserWarning)
                    self.scheduler.step()
                self.global_step += 1
                metrics.setdefault("prediction_loss", float("nan"))
                metrics.setdefault("sigreg_loss", float("nan"))
                metrics.setdefault("total_loss", float("nan"))
                metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
                return metrics
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # Guard against non-finite gradients: an inf/nan grad poisons the
            # optimizer state (exp_avg, exp_avg_sq) and turns the *next* forward
            # into NaN. Detect and skip the step before the damage is done.
            grads_finite = all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in self.model.parameters()
                if p.grad is not None
            )
            if not grads_finite:
                nan_param_count = sum(
                    1 for p in self.model.parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                )
                logger.warning(
                    "Step %d: %d params with non-finite grad, skipping update",
                    self.global_step,
                    nan_param_count,
                )
                self.optimizer.zero_grad()
                with warnings.catch_warnings():
                    # Suppress "lr_scheduler.step() before optimizer.step()"
                    # warning: this path intentionally skips opt.step.
                    warnings.simplefilter("ignore", UserWarning)
                    self.scheduler.step()
                self.global_step += 1
                metrics.setdefault("prediction_loss", float("nan"))
                metrics.setdefault("sigreg_loss", float("nan"))
                metrics.setdefault("total_loss", float("nan"))
                metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
                return metrics
            self.optimizer.step()

        self.scheduler.step()
        self.global_step += 1

        metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
        return metrics

    def train(self) -> None:
        """Run the full training loop."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        init_wandb(self.config)

        logger.info("Starting training from step %d", self.global_step)

        while self.global_step < self.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.max_steps:
                    break

                frames = batch["frames"]
                actions = batch["actions"]

                metrics = self._training_step(frames, actions)

                # Logging
                if self.global_step % self.log_interval == 0:
                    logger.info(
                        "Step %d | prediction_loss=%.4f | sigreg_loss=%.4f"
                        " | total_loss=%.4f | lr=%.6f",
                        self.global_step,
                        metrics.get("prediction_loss", 0),
                        metrics.get("sigreg_loss", 0),
                        metrics.get("total_loss", 0),
                        metrics.get("learning_rate", 0),
                    )
                    log_metrics(metrics, self.global_step)

                # Checkpointing
                if self.global_step % self.checkpoint_interval == 0:
                    ckpt_path = self.output_dir / f"checkpoint_{self.global_step}.pt"
                    save_checkpoint(
                        ckpt_path,
                        self.model,
                        self.optimizer,
                        self.global_step,
                        self.config,
                        scheduler=self.scheduler,
                    )
                    logger.info("Saved checkpoint at step %d", self.global_step)

        # Final checkpoint
        ckpt_path = self.output_dir / f"checkpoint_{self.global_step}.pt"
        save_checkpoint(
            ckpt_path,
            self.model,
            self.optimizer,
            self.global_step,
            self.config,
            scheduler=self.scheduler,
        )
        logger.info(
            "Training complete. Final checkpoint saved at step %d",
            self.global_step,
        )

    def resume(self, checkpoint_path: str | Path) -> None:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        self.global_step = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
        )
        logger.info("Resumed from checkpoint at step %d", self.global_step)
