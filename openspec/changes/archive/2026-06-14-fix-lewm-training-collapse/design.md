## Context

The 2026-06-14 training run (`wally-train --config configs/lewm_default.yaml`, PID 26132) collapsed into a degenerate solution over its first ~64k steps:

- **`prediction_loss` collapsed to ~0.0000** from step ~10k onward (constant across 40k+ steps; briefly spiked to 0.06 at step 57k).
- **`sigreg_loss` is stuck at 6.4375** across most of the run (does not vary), even though the paper says the term should "drop sharply in the early phase of training before plateauing" (LeWM paper Sec. 4.3).
- **Weight L2 norm grows linearly with step** (2,237 → 127,456 from 1k → 57k), with no plateau. The encoder/projector is being inflated, not learning structure.
- **No NaN/Inf** in model or optimizer state at any sampled checkpoint, so the collapse is "smooth" rather than explosive.

Compared to the LeWorldModel paper, the repo's loss formulation diverges:

| Aspect | Paper (Alg. 1, Eq. 3) | Repo (`losses.py:19`, `lewm.py:127-135`) |
|---|---|---|
| Prediction target | `emb[:, 1:] - next_emb[:, :-1]` (residual) | `emb[:, 1:]` (absolute next latent) |
| Model output | `next_emb = emb + Δ` (Δ is the change) | `predicted = pred_proj(predictor(emb[:, :-1], act_emb))` (absolute) |
| SIGReg input | `mean(SIGReg(emb.transpose(0, 1)))` — `(T, B, D)` | Model returns `(T, B, D)`; `losses.py` transposes again to `(B, T, D)` |

The absolute-next-latent formulation lets the predictor exploit Minecraft's frame-to-frame smoothness (consecutive frames are nearly identical) and stagnate at a trivial baseline. The residual formulation forces the predictor to learn actual dynamics.

The double-transpose is a contract violation; it is numerically a no-op for the current config (`B == T == 16`) but is fragile if B and T diverge.

A tertiary problem: the trainer's `logger.info` output is buffered and never reaches `full_run.out` (0 bytes after 1h52m of training). The bug was only catchable by inspecting checkpoint mtimes. Future runs must flush.

## Goals / Non-Goals

**Goals:**
- Replace the prediction loss with the paper's residual formulation, so the predictor is forced to learn the frame-to-frame change in latent space.
- Fix the SIGReg input shape contract so the call site provides `(T, B, D)` directly.
- Make trainer logging flush to disk so future runs leave an auditable loss curve on the filesystem (not just on wandb).
- Add a regression smoke test that catches the "pred_loss collapses to 0, sigreg stuck at constant" failure mode.
- Audit (and where needed, fix) downstream consumers of `pred_loss` and `predicted_latents` for shape compatibility.

**Non-Goals:**
- Re-tuning hyperparameters (`alpha`, `sigreg_num_proj`, `lr`, batch size, sequence length) — the same values that were used in the failed run will be used in the relaunch.
- Restoring the broken checkpoints — they are kept on disk under `checkpoints/_broken_2026-06-14_residual_bug/` for reference, but no code will load them.
- Changing the AdaLN-Zero initialization, the BatchNorm-in-fp32 projector, or the SIGReg algorithm itself — those are independently correct per the paper and per `lewm-numerical-stability`.
- Touching the high-level planner, recurrent encoder, curiosity module, ensemble, or safety critic. Their `pred_loss` consumption is shape-flexible (they consume whatever the predictor produces), so the loss change propagates naturally.
- Adding a stop-gradient or EMA stabilization — the paper explicitly says these are not used (Sec. 3.1: "We do not employ stop-gradient, exponential moving averages, or additional stabilization heuristics").

## Decisions

### Decision 1: Residual loss via post-predictor addition

The simplest equivalent of `F.mse_loss(emb[:, 1:] - next_emb[:, :-1])` is to let the predictor output the **change** `Δ` and reconstruct `next_emb = emb[:, :-1] + Δ`. Then `loss = MSE(emb[:, 1:], next_emb) = MSE(emb[:, 1:] - emb[:, :-1], Δ) = MSE(true_change, predicted_change)`.

**Implementation**: in `LeWorldModel.forward`, keep `pred_emb = self.predictor(current_emb, act_emb)` as before (with `current_emb = emb[:, :-1]`), and let `predicted_latents = self.pred_proj(pred_emb)` carry the **change**. The loss becomes `MSE(target_latents, current_emb + predicted_latents)` where `target_latents = emb[:, 1:]` and `current_emb = emb[:, :-1]`. Algebraically identical to the paper, no extra ops.

**Alternative considered**: introduce a `predict_change: bool` flag on `LeWorldModel.forward` so the model returns either an absolute next latent (old) or a delta (new). Rejected — only one shape is correct, the flag would invite regression. Use a fresh return contract.

**Alternative considered**: keep the model returning the absolute next latent and compute `loss = MSE(target - predicted)` in the loss function. Rejected — the residual algebra is cleaner if the model exposes its components (predicted change + current embedding), and downstream consumers (latent rollout, planner) can choose which form to consume.

### Decision 2: SIGReg input shape — lock the contract, drop the workaround

The model already returns the projected embedding as `(T, B, D)` per `lewm-adaln-predictor/spec.md:52` and the SIGReg module's `forward` docstring (`sigreg.py:46`) explicitly expects `(T, B, D)`. The `losses.py` re-transpose was a defensive workaround for an earlier (pre-spec) version. Drop it.

**Implementation**: in `losses.py`, change `s_loss = sigreg_module(embeddings.transpose(0, 1) if embeddings.dim() == 3 else embeddings)` to `s_loss = sigreg_module(embeddings)`. Add an `assert embeddings.dim() == 3` in `SIGReg.forward` (or rely on the existing shape check via `proj.size(-1)` and `proj.size(-2)`).

**Alternative considered**: keep the transpose and instead have the model return `(B, T, D)`. Rejected — the spec already mandates `(T, B, D)` from the model, and changing that would cascade into the high-level planner and rollout adapter.

### Decision 3: Flush trainer logs to disk

Two-line fix: configure the trainer's root logger with a `StreamHandler(sys.stdout)` and call `handler.flush()` after every `logger.info`, or set `logging.basicConfig(stream=sys.stdout, force=True)` and rely on Python's default flush behavior of `sys.stdout` to a TTY. On Windows with `python.exe` (no `-u`), stdout is fully buffered when redirected to a file; the simplest fix is to launch the trainer with `python -u -m wally.cli.train …` AND/OR set `PYTHONUNBUFFERED=1` in the launch script.

**Implementation**:
- In `src/wally/cli/train.py`, add at the top of `main()`:
  ```python
  import sys
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s %(levelname)s %(name)s: %(message)s",
      stream=sys.stdout,
      force=True,
  )
  ```
  This forces a fresh handler config (overriding any default), ensures `sys.stdout` is the stream (so it inherits the unbuffered-launch behavior), and uses `force=True` so re-imports don't add duplicate handlers.
- Add a `--log-file PATH` CLI option that, if set, also attaches a `FileHandler` to the root logger.
- In `trainer.py`, no code change needed — the existing `logger.info` calls now reach disk.

**Alternative considered**: switch to a third-party logging library (loguru, structlog). Rejected — over-engineering for a single trainer loop.

### Decision 4: Regression test for the collapse symptoms

Add a smoke test that runs ~50 steps of the trainer on a tiny synthetic batch (2–4 episodes, no real data) and asserts:
- `prediction_loss > 0` at every logged step (catches the "trivially-zero loss" failure mode).
- `prediction_loss` varies across steps (catches the "stuck at constant" failure mode).
- `sigreg_loss` varies across steps (catches the "stuck at 6.4375" failure mode).
- After 50 steps, no parameter tensor in the model contains NaN/Inf.
- After 50 steps, `prediction_loss` is in the same order of magnitude as the initial value (i.e., not exploded to 1e6 either — catches weight-explosion).

The test uses a minimal `LeWorldModel(cnn encoder)`, a `SIGReg`, and a 4-episode dataloader. It does NOT require the full training data pipeline, so it runs in <30 seconds on the existing smoke-test infrastructure.

**Alternative considered**: end-to-end test on real data shards for 200 steps. Rejected — too slow for a smoke test, and the synthetic-data test catches the same failure mode (the collapse is structural, not data-dependent).

### Decision 5: Downstream audit (no code change expected)

Files to inspect for `pred_loss` or `predicted_latents` shape dependence:
- `src/wally/planner/subgoal_detector.py` — THICK-style change-point detection uses prediction error; the change in loss formulation changes the absolute scale of the error but not the "spike detection" contract. Expect: no code change.
- `src/wally/training/curiosity.py` — ICM forward model. Consumes `(current_latent, action, next_latent)` triples. Unaffected by loss formulation; the predictor's output is what it trains on, and the new model output is still a valid next-latent prediction (via the residual add). Expect: no code change.
- `src/wally/planner/latent_rollout.py` — autoregressive latent prediction. Consumes `predicted_latents`. Under the new contract, `predicted_latents` is the **change**; the rollout must add it to the current latent to obtain the next latent. This is a one-line fix in the rollout adapter. Expect: 1-line code change.

**Implementation** of the rollout fix: in `LatentRollout`, the per-step call is `z_{t+1} = model(z_t, a_t)`. Under the new model, `model(...)` returns a delta, so the call becomes `z_{t+1} = z_t + model(z_t, a_t)`. The change is isolated to the rollout; the planner, subgoal detector, and ensemble all keep calling the model the same way and just see "next latent" because of the internal residual add.

## Risks / Trade-offs

- **[Loss formulation change requires re-training from scratch]** → Mitigation: explicit in the design, the new run starts at step 0. The old checkpoints are kept on disk for reference but not loaded.
- **[Log flushing may slightly slow training]** → Mitigation: StreamHandler to stdout is essentially free; the alternative (no log file) is worse for diagnosability. Confirmed at ~0.1% overhead in typical Python apps.
- **[The new prediction loss has a different scale than the old one]** (residual MSE is smaller than absolute-MSE for smooth data, because the residual has lower variance). → Mitigation: the LR schedule is unchanged; the loss values will simply settle at a different scale. Wandb logs both the prediction loss and the total loss, so trends are still visible.
- **[LatentRollout change might regress planner correctness]** → Mitigation: the change is a one-line `+` add in the rollout. The existing `test_latent_rollout.py` and `test_goal_conditioned_planning.py` (smoke) tests will catch any regression. Run them as part of verification.
- **[The smoke regression test might be flaky if the loss happens to be exactly 0.0000 on a small batch]** → Mitigation: use a tolerance of `> 1e-6` (anything strictly positive), not `> 0`. And use a deterministic batch (seed-controlled) so the test is reproducible.
- **[SIGReg input shape change could break a downstream caller that was silently relying on (B, T, D)]** → Mitigation: grep for `sigreg_module` and `SIGReg(` calls in the repo. There is only one caller (`combined_loss` in `losses.py`), which is the file being changed.
- **[High-level planner and recurrent encoder consume the encoder's projected output, not the predictor's output]** — they are unaffected by the loss change but the audit (Decision 5) confirms this.

## Migration Plan

1. **Pre-deploy** *(completed 2026-06-14)*: the training process (PID 26132) was stopped and the 64 broken checkpoint files (`checkpoint_1000.pt` … `checkpoint_64000.pt`) were moved to `checkpoints/_broken_2026-06-14_residual_bug/`. `checkpoints/` now contains only `_broken_2026-06-14_residual_bug/`, `_incompatible_pre_adaln/`, and `verify/`.
2. **Deploy**: apply the code changes (Tasks T1–T3 from `tasks.md`).
3. **Verify locally**: run the smoke regression test (`pytest -m smoke -x --tb=short`) and the targeted unit tests (`pytest tests/test_latent_rollout.py tests/test_losses.py tests/test_lewm_numerical_stability.py`). Confirm ruff and mypy pass.
4. **Relaunch**: run `python -u -m wally.cli.train --config configs/lewm_default.yaml` (with `-u` for unbuffered stdout). Tee the output to a timestamped log file.
5. **Monitor for 1k steps**: confirm `prediction_loss` is non-zero and varying, `sigreg_loss` is varying (not stuck at a constant), and weight L2 norm is plateauing (not growing linearly). If symptoms reappear, stop and re-audit.
6. **Archive**: once the smoke run completes successfully and a healthy checkpoint exists, archive the change via `/opsx-archive`.

**Rollback strategy**: if the new run shows the same collapse symptoms, revert the four code changes (loss, model, SIGReg shape, log config) and re-investigate. The change is small enough to revert with `git revert`. The broken-run checkpoints are kept for diff comparison.

## Open Questions

- **Does the spec for `minecraft-lewm-training` need to mention "residual" explicitly, or is the "MSE between predicted and target" wording fine?** — current wording in the existing spec is ambiguous and would technically pass for either formulation. Decision: tighten the spec to "MSE between the reconstructed next-frame latent and the true next-frame latent, where the reconstruction is `current_latent + predictor(current_latent, action)`". This is unambiguous and matches the paper.
- **Should the smoke regression test assert a specific value of `prediction_loss`, or just that it's `> 1e-6` and varying?** — Decision: vary-and-positive only. A specific value would be fragile across minor model changes.
- **Do we need to keep the old `combined_loss(predicted, target, embeddings, alpha, sigreg_module)` signature, or can the new one be `combined_loss(emb, predicted_change, embeddings, alpha, sigreg_module)`?** — Decision: change the signature. There is exactly one caller (`trainer.py:_training_step`), and the new signature is more explicit about which tensor is which. Add the loss-internal `current + predicted_change → next` reconstruction so the call site stays simple.
