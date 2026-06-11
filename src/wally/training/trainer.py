from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from wally.training.checkpoint import load_checkpoint, save_checkpoint
from wally.training.logging import init_wandb, log_metrics
from wally.training.losses import combined_loss
from wally.training.optimizer import create_optimizer
from wally.training.scheduler import create_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop orchestrator for LeWorldModel with SIGReg."""

    def __init__(
        self,
        model: nn.Module,
        critic: nn.Module,
        train_loader: DataLoader[Any],
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.critic = critic
        self.train_loader = train_loader
        self.config = config

        self.lr: float = config.get("lr", 1e-4)
        self.weight_decay: float = config.get("weight_decay", 1e-5)
        self.warmup_steps: int = config.get("warmup_steps", 1000)
        self.max_steps: int = config.get("max_steps", 100_000)
        self.alpha: float = config.get("alpha", 0.1)
        self.use_amp: bool = config.get("use_amp", False)
        self.checkpoint_interval: int = config.get("checkpoint_interval", 1000)
        self.log_interval: int = config.get("log_interval", 10)
        self.output_dir: Path = Path(config.get("output_dir", "checkpoints"))

        default_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device: torch.device = config.get("device", default_device)

        self.model.to(self.device)
        self.critic.to(self.device)

        self.optimizer: Optimizer = create_optimizer(
            self.model, self.lr, self.weight_decay
        )
        self.critic_optimizer: Optimizer = create_optimizer(
            self.critic, self.lr, self.weight_decay
        )
        self.scheduler = create_scheduler(
            self.optimizer, self.warmup_steps, self.max_steps
        )

        self.scaler: GradScaler | None = GradScaler() if self.use_amp else None
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

        # Forward pass
        if self.use_amp:
            with autocast():
                predicted, target = self.model(frames, actions)
                total_loss, metrics = combined_loss(
                    predicted, target, self.critic, self.alpha
                )
        else:
            predicted, target = self.model(frames, actions)
            total_loss, metrics = combined_loss(
                predicted, target, self.critic, self.alpha
            )

        # Backward pass for main model
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # SIGReg critic update (separate optimizer, adversarial)
        # Re-compute sigreg loss so the critic can minimize it (opposite direction)
        if self.use_amp:
            with autocast():
                with torch.no_grad():
                    pred_detached = predicted.detach()
                    target_detached = target.detach()
                from wally.training.sigreg import sigreg_loss

                critic_loss = sigreg_loss(self.critic, pred_detached, target_detached)
        else:
            with torch.no_grad():
                pred_detached = predicted.detach()
                target_detached = target.detach()
            from wally.training.sigreg import sigreg_loss

            critic_loss = sigreg_loss(self.critic, pred_detached, target_detached)

        self.critic_optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)
        else:
            critic_loss.backward()
            self.critic_optimizer.step()

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
                        "Step %d | prediction_loss=%.4f | sigreg_loss=%.4f | total_loss=%.4f | lr=%.6f",
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
                        self.critic_optimizer,
                        self.global_step,
                        self.config,
                    )
                    logger.info("Saved checkpoint at step %d", self.global_step)

        # Final checkpoint
        ckpt_path = self.output_dir / f"checkpoint_{self.global_step}.pt"
        save_checkpoint(
            ckpt_path,
            self.model,
            self.optimizer,
            self.critic_optimizer,
            self.global_step,
            self.config,
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
            self.critic_optimizer,
        )
        logger.info("Resumed from checkpoint at step %d", self.global_step)
