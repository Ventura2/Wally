## Context

A 100k-step training run of the LeWorldModel was started on a clean checkpoint (`checkpoint_44000.pt`, 0/100 NaN tensors) and resumed from there. Within ~2000 steps of resume, all model parameters were corrupted to NaN, and the corruption persisted for the rest of the 21,500 logged steps before being killed. Verified by direct inspection of `checkpoints/checkpoint_66000.pt` (96/100 NaN tensors) vs `checkpoints/checkpoint_44000.pt` (0/100 NaN). Data was independently verified to be clean (frames in [0,1], actions in [-1,1], no NaN/Inf). The bug is in the training loop, not the data or hardware.

Three independent defects compound to make the run unsalvageable:

1. **Wrong SIGReg algorithm.** The current `SIGRegCritic` + `sigreg_loss` is a MINE-style mutual-information estimator with a `tanh`-bounded critic output. The original LeWorldModel paper uses a closed-form Epps-Pulley statistic on random projections of the embeddings — a stateless, non-negative, unconditionally stable alternative. The current implementation has an adversarial critic that can saturate, has the wrong loss sign for the model objective, and produces loss values in `[-2, 2]` rather than `[0, ∞)`. This is the most likely NaN source.
2. **No NaN guard in the training step.** A single bad batch (or any numerical instability) propagates NaN through all parameters and stays NaN forever — there is no mechanism to skip the bad step and continue.
3. **Scheduler state not saved/restored.** On resume, the `LambdaLR` scheduler re-initializes with `last_epoch=0`, so warmup re-runs for 500 wasted steps and the LR trajectory is discontinuous. Also, BN running statistics are silently re-initialized from a tiny warmup batch, which contributes to instability.

The CNN encoder's BatchNorm running inside `autocast(bfloat16)` is a known anti-pattern that should be addressed alongside the SIGReg fix.

## Goals / Non-Goals

**Goals:**
- Restore numerical stability so a 100k-step run finishes with finite losses and uncorrupted weights
- Match the SIGReg algorithm in the LeWM paper exactly (closed-form Epps-Pulley, stateless)
- Add a NaN guard so a single degenerate batch cannot poison the run
- Make resume work correctly (restore scheduler, restore BN running stats already happens via the model state dict)
- Provide a regression test that catches this class of bug before future long runs
- Keep the public CLI / YAML config surface compatible (additive config only)

**Non-Goals:**
- Re-tuning hyperparameters (`alpha`, LR, batch size, sequence length)
- Switching the encoder from CNN to ViT
- Re-implementing the data pipeline
- Adding a separate critic-based MI estimator as an option (one algorithm, not two)
- Modifying the predictor architecture

## Decisions

### D1. Replace `SIGRegCritic` with the paper's `SIGReg` (closed-form)

**Choice:** Port `module.SIGReg` from `lucas-maes/le-wm/module.py:8-37` directly into `src/wally/training/sigreg.py`. Remove `SIGRegCritic`, the `critic_optimizer`, and the adversarial critic-update block in `trainer.py:_training_step`.

**Rationale:** The closed-form SIGReg computes the Epps-Pulley statistic on `num_proj` random directions of the embedding. The loss is non-negative, differentiable only in the embedding (not in the projection matrix), and has no parameters. It cannot saturate, cannot produce NaN from finite inputs, and matches the paper's algorithm byte-for-byte. Reference implementation:

```python
class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
```

**Wally adaptation:** the loss is computed on the **encoder embeddings** (not on `(predicted, target)`), matching `le-wm/train.py:41-42` where `self.sigreg(emb.transpose(0, 1))`. This requires exposing the encoder embeddings through `LeWorldModel.forward` (return a third output, or add a `return_embeddings=True` flag) so the SIGReg sees the same input the le-wm paper uses.

**Alternatives considered:**
- *Keep the critic but fix the sign / add grad penalty* — rejected, this just papers over the fundamental instability of adversarial training with bounded critics. The closed-form version is simpler, faster, and provably stable.
- *Use NWJ / MINE-f / InfoNCE critics with gradient penalty* — rejected, more complex than the paper's choice and not what the paper reports.

### D2. NaN guard with skip-step semantics

**Choice:** At the start of `_training_step`, after the forward pass, check `torch.isfinite(total_loss).all()`. If false, log a warning with the step number, call `self.optimizer.zero_grad()` and `self.critic_optimizer.zero_grad()` (or the unified optimizer if D1 removes the critic), increment `global_step`, and return the previous-step metrics without applying gradients.

**Rationale:** A bad batch should be skipped, not amplified. We still advance `global_step` so logging/checkpoint cadence is preserved. This is a one-line change with high safety value.

### D3. Persist and restore `scheduler.state_dict()` in checkpoints

**Choice:** Add `"scheduler_state_dict": self.scheduler.state_dict()` to the checkpoint dict in `save_checkpoint`, and call `self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])` in `load_checkpoint` (optional param, default `None`). Also handle the now-removed `critic_optimizer_state_dict` gracefully — if it's missing in older checkpoints, skip with a debug log.

**Rationale:** `LambdaLR` stores `last_epoch` and the current LR in its state. Without restoring, every resume re-runs warmup and breaks the LR schedule.

### D4. Sanitize inputs with `torch.nan_to_num`

**Choice:** In `Trainer._training_step`, after `frames.to(self.device)` and `actions.to(self.device)`, apply `torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)`. The data loader's `actions.clamp(-1, 1)` only catches finite out-of-range values, not NaN/Inf.

**Rationale:** Defense in depth. Matches the le-wm `train.py:32` pattern. Catches the rare case where a malformed shard has NaN actions or extreme pixel values.

### D5. BatchNorm runs in fp32 inside autocast

**Choice:** Wrap the CNN encoder's forward in `@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)`. BN running statistics stay in fp32, but the convs benefit from bf16 matmul accumulation. Alternative: cast inputs to fp32 inside `forward` and cast output back to the autocast dtype.

**Rationale:** BN in bf16 is the documented anti-pattern (PyTorch's own AMP examples cast BN to fp32). This is a one-line change in `cnn_encoder.py`.

### D6. New capability `lewm-numerical-stability`

**Choice:** Add a new spec capturing the stability contracts (finite-loss guarantee, NaN guard, resume correctness) and the regression test that enforces them. This is the surface that other capabilities (planning, evaluation) can rely on.

**Rationale:** The existing `lewm-training-loop` spec describes what the loop *does*; the new spec describes what the loop *guarantees*. Separating these makes the stability invariants explicit and testable.

## Risks / Trade-offs

- **R1: Different regularization surface area.** The closed-form SIGReg is on `encoder embeddings` (192-dim after `t=16` frames × `B=16` batch = 256 vectors of dim 192). The critic-based loss was on `(predicted, target)` pairs. The gradient flows into a different part of the model. → Mitigation: this is what the paper uses; the SIGReg magnitude should now be O(1) and stable, which is the desired behavior.

- **R2: Removing the critic breaks any external caller that imports `SIGRegCritic` or `sigreg_loss`.** → Mitigation: internal-only modules; grep confirms no external imports. `losses.combined_loss` signature changes from `(predicted, target, critic, alpha)` to `(predicted, target, encoder_embeddings, alpha)`, but only `trainer.py` calls it.

- **R3: NaN guard masks real problems.** If the loss is consistently NaN, the guard will skip every step and the run will appear to make no progress. → Mitigation: log the skip rate at `log_interval` so silent failure is visible; the regression test asserts loss is finite after 50 steps on synthetic data, catching the most common regressions.

- **R4: BN fp32 inside autocast may be slightly slower.** → Mitigation: minor cost vs the stability win; can be reverted if profiling shows it's a bottleneck. BN layers are only 3 small 2D BNs in the CNN encoder.

- **R5: Older checkpoints with `critic_optimizer_state_dict` may fail to load.** → Mitigation: `load_checkpoint` uses `.get("critic_optimizer_state_dict")` and skips if absent; logs an info message.

- **R6: Resume from a pre-fix checkpoint will load the saved LR-schedule-frozen state.** → Mitigation: if `scheduler_state_dict` is missing (old ckpt), warn and continue with fresh scheduler at the correct `last_epoch = global_step - 1`. The optimizer and model weights load fine.

## Migration Plan

1. Land the code changes behind a passing `pytest -m smoke` run
2. Manually verify the new SIGReg loss is `O(1)` (≈ 0.5–2.0) on a small batch
3. Resume training from `checkpoint_44000.pt` (the last clean checkpoint) — this is the recommended restart point
4. Archive the corrupted `checkpoint_45000.pt`–`checkpoint_66000.pt` files (or move to `checkpoints/corrupted/`) to avoid accidental resume
5. Run a 100-step smoke training on synthetic data to confirm the regression test passes
6. Kick off the new 100k run with the fixed code

**Rollback:** If the new SIGReg produces unstable training for a reason not predicted, revert the change via `git revert` of the merge commit. The new checkpoint format is backwards-compatible (extra `scheduler_state_dict` key, optional `critic_optimizer_state_dict`).

## Open Questions

- Should the `alpha` default change from 0.1 to something else now that SIGReg has a different magnitude? The paper's published configs use 0.1, so keep at 0.1 for parity.
- Should the regression test use a fixed random seed for determinism? — Yes, `torch.manual_seed(0)` at the top of the test.
- Should we add a `model.eval()` shortcut in the smoke test to also verify the eval-mode forward is finite? — Out of scope for this change; can be added in a follow-up.
