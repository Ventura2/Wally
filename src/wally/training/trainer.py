from __future__ import annotations

import logging
import time
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
        *,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.sigreg = sigreg
        self.train_loader = train_loader
        self.config = config
        self._model_config = model_config

        self.lr: float = config.get("lr", 1e-4)
        self.weight_decay: float = config.get("weight_decay", 1e-5)
        self.warmup_steps: int = config.get("warmup_steps", 1000)
        self.max_steps: int = config.get("max_steps", 100_000)
        self.alpha: float = config.get("alpha", 0.1)
        self.vicreg_weight: float = config.get("vicreg_weight", 0.0)
        self.vicreg_std_target: float = config.get("vicreg_std_target", 1.0)
        self.vicreg_cov_weight: float = config.get("vicreg_cov_weight", 1.0)
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
        self.early_stop: bool = config.get("early_stop", False)
        self.early_stop_patience: int = config.get("early_stop_patience", 500)
        self.early_stop_min_step: int = config.get("early_stop_min_step", 1000)
        self.early_stop_ema_alpha: float = config.get("early_stop_ema_alpha", 0.1)
        self.early_stop_min_delta: float = config.get("early_stop_min_delta", 0.0)
        self._ema_total_loss: float | None = None
        self._best_ema_total_loss: float = float("inf")
        self._best_step: int = 0
        self._steps_since_best: int = 0
        self._stop_training: bool = False

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
        frames = frames.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        frames = torch.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward pass
        if self.use_amp:
            with autocast("cuda", dtype=self.amp_dtype):
                predicted_change, emb_T_B_D = self.model(
                    frames, actions, return_embeddings=True
                )
                # emb_T_B_D is (T, B, D); residual loss needs (B, T, D)
                emb_B_T_D = emb_T_B_D.transpose(0, 1)
                total_loss, metrics = combined_loss(
                    emb_B_T_D,
                    predicted_change,
                    emb_T_B_D,
                    self.alpha,
                    self.sigreg,
                    vicreg_weight=self.vicreg_weight,
                    vicreg_std_target=self.vicreg_std_target,
                    vicreg_cov_weight=self.vicreg_cov_weight,
                )
        else:
            predicted_change, emb_T_B_D = self.model(
                frames, actions, return_embeddings=True
            )
            emb_B_T_D = emb_T_B_D.transpose(0, 1)
            total_loss, metrics = combined_loss(
                emb_B_T_D,
                predicted_change,
                emb_T_B_D,
                self.alpha,
                self.sigreg,
                vicreg_weight=self.vicreg_weight,
                vicreg_std_target=self.vicreg_std_target,
                vicreg_cov_weight=self.vicreg_cov_weight,
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
            if self.vicreg_weight > 0.0:
                metrics.setdefault("vicreg_loss", float("nan"))
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
                if self.vicreg_weight > 0.0:
                    metrics.setdefault("vicreg_loss", float("nan"))
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
                if self.vicreg_weight > 0.0:
                    metrics.setdefault("vicreg_loss", float("nan"))
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
        run_name = f"{self.config['wandb_project']}-step-{self.global_step}"
        init_wandb(self.config, name=run_name)

        logger.info("Starting training from step %d", self.global_step)
        if self.early_stop:
            logger.info(
                "Early stop: patience=%d, min_step=%d, ema_alpha=%.2f, min_delta=%.4f",
                self.early_stop_patience,
                self.early_stop_min_step,
                self.early_stop_ema_alpha,
                self.early_stop_min_delta,
            )

        last_log_t = time.perf_counter()
        while self.global_step < self.max_steps and not self._stop_training:
            for batch in self.train_loader:
                if self.global_step >= self.max_steps:
                    break

                fetch_t = time.perf_counter()

                frames = batch["frames"]
                actions = batch["actions"]

                metrics = self._training_step(frames, actions)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_t = time.perf_counter()

                fetch_s = fetch_t - last_log_t
                gpu_s = step_t - fetch_t
                total_s = step_t - last_log_t
                last_log_t = step_t

                # Logging
                if self.global_step % self.log_interval == 0:
                    vicreg_str = ""
                    vicreg_args: tuple[float, ...] = ()
                    if "vicreg_loss" in metrics:
                        vicreg_str = " | vicreg_loss=%.4f"
                        vicreg_args = (metrics["vicreg_loss"],)
                    logger.info(
                        "Step %d | prediction_loss=%.4f | sigreg_loss=%.4f%s"
                        " | total_loss=%.4f | lr=%.6f"
                        " | fetch=%.2fs gpu=%.3fs total=%.2fs",
                        self.global_step,
                        metrics.get("prediction_loss", 0),
                        metrics.get("sigreg_loss", 0),
                        vicreg_str,
                        *vicreg_args,
                        metrics.get("total_loss", 0),
                        metrics.get("learning_rate", 0),
                        fetch_s,
                        gpu_s,
                        total_s,
                    )
                    log_metrics(metrics, self.global_step)

                if self.early_stop:
                    self._update_early_stop(metrics.get("total_loss", 0.0))

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
                        model_config=self._model_config,
                    )
                    logger.info("Saved checkpoint at step %d", self.global_step)

                if self._stop_training:
                    break

        # Final checkpoint
        ckpt_path = self.output_dir / f"checkpoint_{self.global_step}.pt"
        save_checkpoint(
            ckpt_path,
            self.model,
            self.optimizer,
            self.global_step,
            self.config,
            scheduler=self.scheduler,
            model_config=self._model_config,
        )
        if self._stop_training:
            logger.info(
                "Training stopped early at step %d (best EMA total_loss=%.4f at step %d, "
                "patience=%d, min_step=%d). Use checkpoint_best.pt for the best weights.",
                self.global_step,
                self._best_ema_total_loss,
                self._best_step,
                self.early_stop_patience,
                self.early_stop_min_step,
            )
        else:
            logger.info(
                "Training complete. Final checkpoint saved at step %d",
                self.global_step,
            )

    def _update_early_stop(self, total_loss: float) -> None:
        if self.global_step < self.early_stop_min_step:
            return
        if self._ema_total_loss is None:
            self._ema_total_loss = float(total_loss)
        else:
            alpha = self.early_stop_ema_alpha
            self._ema_total_loss = alpha * float(total_loss) + (1.0 - alpha) * self._ema_total_loss
        if self._ema_total_loss < self._best_ema_total_loss - self.early_stop_min_delta:
            self._best_ema_total_loss = self._ema_total_loss
            self._best_step = self.global_step
            self._steps_since_best = 0
            best_path = self.output_dir / "checkpoint_best.pt"
            save_checkpoint(
                best_path,
                self.model,
                self.optimizer,
                self.global_step,
                self.config,
                scheduler=self.scheduler,
                model_config=self._model_config,
            )
            logger.info(
                "New best EMA total_loss=%.4f at step %d (saved %s)",
                self._best_ema_total_loss,
                self.global_step,
                best_path,
            )
        else:
            self._steps_since_best += 1
        if self._steps_since_best >= self.early_stop_patience:
            logger.info(
                "Early-stop trigger: %d steps since best (patience=%d). "
                "Best EMA total_loss=%.4f at step %d.",
                self._steps_since_best,
                self.early_stop_patience,
                self._best_ema_total_loss,
                self._best_step,
            )
            self._stop_training = True

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
