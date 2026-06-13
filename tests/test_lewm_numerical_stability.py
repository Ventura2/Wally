from __future__ import annotations

from itertools import cycle
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from wally.data.dataloader import create_dataloader
from wally.models.lewm import LeWorldModel
from wally.models.lewm_blocks import ConditionalBlock
from wally.training.checkpoint import save_checkpoint
from wally.training.sigreg import SIGReg
from wally.training.trainer import Trainer


class _SyntheticIterableDataset(IterableDataset):
    def __init__(self, num_batches: int = 4) -> None:
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                "frames": torch.randint(0, 256, (2, 8, 3, 64, 64), dtype=torch.uint8)
                .float()
                .div_(255.0),
                "actions": torch.clamp(torch.randn(2, 8, 25), -1.0, 1.0),
            }


def _identity_collate(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return batch


def _make_trainer(
    config: dict | None = None,
    loader_batches: int = 4,
) -> Trainer:
    if config is None:
        config = {
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "warmup_steps": 500,
            "max_steps": 100_000,
            "alpha": 0.01,
            "use_amp": False,
            "device": "cpu",
        }
    model = LeWorldModel(
        encoder_type="cnn", embed_dim=64, depth=2, num_heads=4
    )
    sigreg = SIGReg(knots=9, num_proj=64)
    loader = DataLoader(
        _SyntheticIterableDataset(loader_batches),
        batch_size=None,
        collate_fn=_identity_collate,
    )
    return Trainer(model, sigreg, loader, config)


class TestLeWMNumericalStability:
    @pytest.mark.smoke
    def test_sigreg_finite_on_normal_embeddings(self):
        torch.manual_seed(0)
        sigreg = SIGReg()
        embeddings = torch.randn(16, 4, 192)
        out = sigreg(embeddings)
        assert torch.isfinite(out)
        assert out.item() >= 0.0
        assert out.item() < 5.0

    @pytest.mark.smoke
    def test_sigreg_finite_on_zeros(self):
        sigreg = SIGReg()
        embeddings = torch.zeros(16, 4, 192)
        out = sigreg(embeddings)
        assert torch.isfinite(out)
        assert not torch.isnan(out)
        assert not torch.isinf(out)
        assert out.item() >= 0.0

    @pytest.mark.smoke
    def test_sigreg_finite_on_constant(self):
        sigreg = SIGReg()
        embeddings = torch.full((16, 4, 192), 3.14)
        out = sigreg(embeddings)
        assert torch.isfinite(out)
        assert not torch.isnan(out)
        assert not torch.isinf(out)
        assert out.item() >= 0.0

    @pytest.mark.smoke
    def test_sigreg_gradients_flow_to_embedding(self):
        torch.manual_seed(0)
        sigreg = SIGReg()
        embeddings = torch.randn(16, 4, 192, requires_grad=True)
        out = sigreg(embeddings)
        out.backward()
        assert embeddings.grad is not None
        assert torch.isfinite(embeddings.grad).all()

    @pytest.mark.smoke
    def test_sigreg_has_no_learnable_params(self):
        sigreg = SIGReg()
        assert len(list(sigreg.parameters())) == 0

    @pytest.mark.smoke
    def test_smoke_run_finite_loss(self):
        torch.manual_seed(0)
        trainer = _make_trainer()
        for i, batch in enumerate(cycle(trainer.train_loader)):
            if i >= 50:
                break
            metrics = trainer._training_step(batch["frames"], batch["actions"])
            assert torch.isfinite(
                torch.tensor(metrics.get("total_loss", float("nan")))
            ), f"non-finite total_loss at step {i+1}: {metrics.get('total_loss')}"

    @pytest.mark.smoke
    def test_smoke_run_checkpoint_no_nan(self, tmp_path):
        torch.manual_seed(0)
        trainer = _make_trainer()
        for i, batch in enumerate(cycle(trainer.train_loader)):
            if i >= 50:
                break
            trainer._training_step(batch["frames"], batch["actions"])

        ckpt_path = tmp_path / "smoke_no_nan.pt"
        save_checkpoint(
            ckpt_path,
            trainer.model,
            trainer.optimizer,
            trainer.global_step,
            trainer.config,
            scheduler=trainer.scheduler,
        )

        payload = torch.load(ckpt_path, weights_only=False)
        for name, tensor in payload["model_state_dict"].items():
            assert torch.isfinite(tensor).all(), f"NaN/Inf in {name}"

    @pytest.mark.smoke
    def test_nan_guard_skips_step(self):
        torch.manual_seed(0)
        trainer = _make_trainer()

        original_forward = trainer.model.forward
        call_count = {"n": 0}

        def patched_forward(*args, **kwargs):
            call_count["n"] += 1
            predicted, target, embeddings = original_forward(*args, **kwargs)
            if call_count["n"] == 2:
                predicted = torch.full_like(predicted, float("nan"))
            return predicted, target, embeddings

        trainer.model.forward = patched_forward

        batches = []
        for i, batch in enumerate(cycle(trainer.train_loader)):
            batches.append(batch)
            if len(batches) >= 3:
                break

        metrics_step1 = trainer._training_step(
            batches[0]["frames"], batches[0]["actions"]
        )
        param_snapshot = {
            name: p.detach().clone()
            for name, p in trainer.model.named_parameters()
        }

        metrics_step2 = trainer._training_step(
            batches[1]["frames"], batches[1]["actions"]
        )

        for name, param in trainer.model.named_parameters():
            assert torch.equal(param, param_snapshot[name]), (
                f"learnable param {name} changed on NaN step"
            )

        metrics_step3 = trainer._training_step(
            batches[2]["frames"], batches[2]["actions"]
        )

        assert torch.isfinite(
            torch.tensor(metrics_step1.get("total_loss", float("nan")))
        )
        assert torch.isnan(
            torch.tensor(metrics_step2.get("total_loss", float("nan")))
        )
        assert torch.isfinite(
            torch.tensor(metrics_step3.get("total_loss", float("nan")))
        )

        assert trainer.global_step == 3
        trainer.model.forward = original_forward

    @pytest.mark.smoke
    def test_resume_does_not_rewarmup(self, tmp_path):
        torch.manual_seed(0)
        config = {
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "warmup_steps": 500,
            "max_steps": 100_000,
            "alpha": 0.01,
            "use_amp": False,
            "device": "cpu",
        }
        trainer1 = _make_trainer(config=config)
        for _ in range(9999):
            trainer1.scheduler.step()
        trainer1.global_step = 10_000

        ckpt_path = tmp_path / "resume.pt"
        save_checkpoint(
            ckpt_path,
            trainer1.model,
            trainer1.optimizer,
            trainer1.global_step,
            trainer1.config,
            scheduler=trainer1.scheduler,
        )

        trainer2 = _make_trainer(config=config)
        assert trainer2.scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-8)

        trainer2.resume(ckpt_path)

        assert trainer2.global_step == 10_000
        assert trainer2.scheduler.last_epoch == 9_999

        lr_after_resume = trainer2.scheduler.get_last_lr()[0]
        assert lr_after_resume > 1e-5
        assert lr_after_resume < 1e-4

        import math

        progress = (10_000 - 500) / (100_000 - 500)
        expected_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        expected_lr = 1e-4 * expected_factor
        assert lr_after_resume == pytest.approx(expected_lr, rel=1e-4)

        trainer2.scheduler.step()
        assert trainer2.scheduler.last_epoch == 10_000
        lr_after_step = trainer2.scheduler.get_last_lr()[0]
        assert lr_after_step < lr_after_resume

    @pytest.mark.smoke
    def test_input_nan_sanitized(self):
        torch.manual_seed(0)
        trainer = _make_trainer()
        batch = next(iter(_SyntheticIterableDataset(1)))
        frames = batch["frames"].clone()
        actions = batch["actions"].clone()
        actions[0, 0, 0] = float("nan")
        frames[0, 0, 0, 0, 0] = float("inf")

        metrics = trainer._training_step(frames, actions)
        assert torch.isfinite(
            torch.tensor(metrics.get("total_loss", float("nan")))
        )

    @pytest.mark.smoke
    def test_adaln_modulation_zero_at_init(self):
        """AdaLN-Zero modulation is zero-initialized so the ConditionalBlock is
        a strict identity at step 0.

        The modulation is a single Linear(c_dim, 6*dim) that produces
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        With zero weight and zero bias, the gate chunks (positions 2 and 5
        of the 6-tuple) are exactly zero for any input, so the attention
        and MLP contributions are scaled to zero and the block returns its
        input unchanged. This is what makes the predictor numerically
        stable from the first step.
        """
        torch.manual_seed(0)
        cb = ConditionalBlock(dim=64, num_heads=4, mlp_ratio=4.0, c_dim=64)

        assert torch.all(cb.modulation.weight == 0), (
            "modulation.weight must be all zeros at init (AdaLN-Zero)"
        )
        assert torch.all(cb.modulation.bias == 0), (
            "modulation.bias must be all zeros at init (AdaLN-Zero)"
        )

        x = torch.randn(2, 8, 64)
        c = torch.randn(2, 8, 64)
        chunks = cb.modulation(F.silu(c)).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        assert torch.all(gate_msa == 0), (
            "gate_msa (chunk 2) must be exactly zero for any input when "
            "the modulation linear is zero-initialized"
        )
        assert torch.all(gate_mlp == 0), (
            "gate_mlp (chunk 5) must be exactly zero for any input when "
            "the modulation linear is zero-initialized"
        )

        out = cb(x, c)
        assert torch.allclose(out, x), (
            "ConditionalBlock must be a strict identity at init when the "
            "modulation linear is zero-initialized"
        )


# ---------------------------------------------------------------------------
# End-to-end stability regression: real data, real config, real GPU.
#
# This is the test that would have caught the failure mode observed on
# 2026-06-13: a freshly-init LeWorldModel with the production config
# (batch_size=16, bf16, CNN encoder, depth=4, embed_dim=192, warmup=500)
# produced non-finite gradients within a handful of optimizer steps,
# poisoning the weights. The test guards against the same class of bug
# by running 100 trainer steps on real Minecraft shards and asserting
# every parameter stays finite, even on the steps that get skipped by
# the grad guard.
# ---------------------------------------------------------------------------

_REAL_SHARDS = Path("data") / "shards" / "chunks"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA GPU (TheRock multi-arch PyTorch on RX 6700 XT)",
)
@pytest.mark.skipif(
    not _REAL_SHARDS.is_dir() or not any(_REAL_SHARDS.glob("*.tar")),
    reason=f"requires real Minecraft shards in {_REAL_SHARDS}",
)
class TestLeWMRealDataStability:
    """End-to-end trainer on real data, real config, real GPU."""

    def _make_production_trainer(self) -> Trainer:
        from wally.config.loader import load_config

        train_config, model_config = load_config(Path("configs/lewm_default.yaml"))
        device = torch.device("cuda")
        model = LeWorldModel(
            vit_variant=model_config.vit_variant,
            embed_dim=model_config.embed_dim,
            depth=model_config.depth,
            num_heads=model_config.num_heads,
            mlp_ratio=model_config.mlp_ratio,
            dropout=model_config.dropout,
            action_dim=model_config.action_dim,
            pretrained=model_config.pretrained,
            encoder_type=model_config.encoder_type,
        ).to(device)
        sigreg = SIGReg(
            num_proj=train_config.sigreg_num_proj, knots=train_config.sigreg_knots
        ).to(device)
        dataloader = create_dataloader(
            data_dir=str(train_config.data_dir),
            batch_size=train_config.batch_size,
            num_workers=2,
            seq_length=train_config.seq_length,
            skip_short=train_config.skip_short,
            persistent_workers=False,
            prefetch_factor=2,
        )
        config_dict = train_config.to_dict()
        config_dict["device"] = device
        return Trainer(model, sigreg, dataloader, config_dict)

    @pytest.mark.smoke
    def test_200_steps_real_data_no_nan_params(self):
        """200 trainer steps on real Minecraft shards; no parameter goes NaN/Inf.

        This is the test that would have failed on 2026-06-13 with the
        ReZero-less predictor and the no-grad-guard trainer. With the fix
        (ReZero init + TransformerEncoder + grad guard), every step either
        updates or is skipped, but the model state stays clean.
        """
        torch.manual_seed(0)
        trainer = self._make_production_trainer()
        skipped = 0
        for i, batch in enumerate(trainer.train_loader):
            if i >= 200:
                break
            metrics = trainer._training_step(batch["frames"], batch["actions"])
            if not torch.isfinite(
                torch.tensor(metrics.get("total_loss", float("nan")))
            ):
                skipped += 1
            # CRITICAL: even on skipped steps, no param may be NaN/Inf
            for n, p in trainer.model.named_parameters():
                assert torch.isfinite(p).all(), (
                    f"non-finite param at step {i+1}: {n} "
                    f"(total_loss={metrics.get('total_loss')}, skipped={skipped})"
                )
        # Sanity: the model should have actually trained on almost every step
        # (allow up to ~2.5% skips; the grad guard's job is to never poison)
        assert skipped <= 5, (
            f"too many skipped steps ({skipped}/200) — grad guard is "
            f"hiding a deeper instability"
        )

    @pytest.mark.smoke
    def test_200_steps_real_data_losses_finite_mostly(self):
        """200 trainer steps on real data; at least 80% of steps are finite.

        The grad guard is allowed to skip a few bad batches, but the
        overwhelming majority should produce finite losses — if not,
        the model is fundamentally broken.
        """
        torch.manual_seed(0)
        trainer = self._make_production_trainer()
        finite_count = 0
        total = 0
        for i, batch in enumerate(trainer.train_loader):
            if i >= 200:
                break
            metrics = trainer._training_step(batch["frames"], batch["actions"])
            total += 1
            if torch.isfinite(
                torch.tensor(metrics.get("total_loss", float("nan")))
            ):
                finite_count += 1
        assert total > 0, "dat loader produced no batches"
        assert finite_count >= int(0.8 * total), (
            f"only {finite_count}/{total} steps produced finite losses — "
            f"model is broken"
        )
