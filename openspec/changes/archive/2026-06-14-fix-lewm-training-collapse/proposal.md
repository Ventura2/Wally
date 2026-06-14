## Why

The LeWorldModel training run launched on 2026-06-14 collapsed into a degenerate solution: `prediction_loss ≈ 0.0000` from step ~10k onward, `sigreg_loss` flat at 6.4375 for 40k+ steps, and weight L2 norm growing linearly with step (2,237 → 127,456 across 1k→57k checkpoints). The model is not learning dynamics — the predictor is being driven to a trivial baseline by exploiting Minecraft frame-to-frame temporal smoothness, and the encoder/projector is being inflated to satisfy SIGReg (which it never actually satisfies).

The root cause is a divergence from the LeWorldModel paper (Algorithm 1, Eq. 3): the paper's prediction loss is on the **residual** `F.mse_loss(emb[:, 1:] - next_emb[:, :-1])` (model predicts the change, target is the difference), but the current implementation computes `F.mse_loss(predicted, emb[:, 1:])` (model predicts the absolute next latent). With AdaLN-Zero initialization, both formulations start at the same point, but the residual form forces the predictor to learn actual dynamics; the absolute form lets it exploit smoothness and stagnate.

A secondary bug: the SIGReg input is transposed twice (model returns (T, B, D), `losses.py` transposes again to (B, T, D)). Numerically a no-op for the current config (`B == T == 16` so `proj.size(-2)` is the same), but breaks the spec contract and could silently change the statistic if B and T diverge.

A tertiary problem surfaced during diagnosis: the trainer's `logger.info` output is buffered and never reaches `full_run.out` (0 bytes after 1h52m of training), so loss curves are only available via checkpoint timestamps. Future runs must flush logs.

This blocks everything downstream that consumes `pred_loss` or the encoder's latents: `subgoal-detection` (THICK uses prediction-error spikes), `curiosity-exploration` (ICM uses prediction error as intrinsic reward), `high-level-planner` (shares the encoder), `memory-augmentation` (LSTM over the encoder), and `safety-critic` (ensemble variance assumes models differ meaningfully).

## What Changes

- Replace the prediction loss in `combined_loss` with the paper's residual formulation: `F.mse_loss(emb[:, 1:] - next_emb[:, :-1])`. The model's predictor output is interpreted as a per-step change, not an absolute next latent.
- Adjust the model's `forward` return contract so `predicted_latents` carries the **change** prediction (or equivalently, the model adds the prediction to the current embedding before returning), making the loss a direct MSE between predicted change and true change.
- Remove the redundant `embeddings.transpose(0, 1)` in `losses.py`. SIGReg now receives `(T, B, D)` as the spec requires, no double-transpose.
- Configure the trainer's logger to flush after every `logger.info` so loss values are recoverable from disk, not just wandb.
- Add a `--log-file` option (or hardcode a tee-style handler) so stdout/stderr from `wally-train` reach `full_run.out` line-by-line, even if the user doesn't run with `python -u`.
- Add a smoke test that runs ~100 steps on a tiny synthetic batch, asserts `prediction_loss > 0` and finite, asserts `sigreg_loss` varies across steps (not stuck at a constant), and dumps the loss curve for visual inspection.
- Audit downstream consumers (`subgoal_detector.py`, `curiosity.py`, `latent_rollout.py`) for any code that depends on the absolute-next-latent shape of `predicted`; document or fix any such code.

## Capabilities

### New Capabilities

(none — this is a bugfix that re-aligns existing capabilities with the paper)

### Modified Capabilities

- `lewm-training-loop`: prediction loss target changes from absolute next latent to residual (frame-to-frame change). SIGReg input shape contract changes from any 3D to explicit `(T, B, D)`. Logger must flush.
- `lewm-adaln-predictor`: the predictor's output semantics change — it is no longer "next latent in absolute terms" but "delta to add to current latent to obtain the next latent". The model's `forward` return contract is updated to match.
- `lewm-numerical-stability`: the SIGReg module's input shape is locked to `(T, B, D)`. The double-transpose workaround is removed; the call site must provide the correct shape directly.
- `minecraft-lewm-training`: top-level training spec inherits the residual-loss change and the flush-on-log requirement. The observed "training collapse" symptoms are explicitly added as a regression scenario.

## Impact

- **Code**:
  - `src/wally/training/losses.py` — `combined_loss` signature/return contract
  - `src/wally/models/lewm.py` — `forward` return semantics for `predicted_latents` (now `emb[:, :-1] + predictor(...)` or equivalent)
  - `src/wally/training/sigreg.py` — input-shape docstring/assertion
  - `src/wally/cli/train.py` — logger config (flush handlers, optional `--log-file`)
  - `src/wally/training/trainer.py` — `logger.info` calls remain, but inherit the new flush handler
- **Downstream consumers to audit (not necessarily change)**:
  - `src/wally/planner/subgoal_detector.py` — uses prediction error to detect change points; behavior changes with the new loss shape but the "use prediction error" contract is unchanged
  - `src/wally/training/curiosity.py` — ICM forward model loss; check whether it consumes the same predicted/target pair shape
  - `src/wally/planner/latent_rollout.py` — uses predicted latents in autoregressive rollouts; needs the new "delta to add" semantics
  - `src/wally/planner/high_level_planner.py` — uses the encoder's projected output (not the predictor), so should be unaffected
- **Tests**:
  - `tests/test_lewm_numerical_stability.py` — add a regression test for the residual loss producing a non-zero, decreasing value
  - `tests/test_losses.py` (or new) — assert `combined_loss` shape and that `sigreg` receives `(T, B, D)` not `(B, T, D)`
- **Data / artifacts**:
  - The 64 broken checkpoints (`checkpoint_1000.pt` … `checkpoint_64000.pt`) were moved to `checkpoints/_broken_2026-06-14_residual_bug/` on 2026-06-14 (not deleted — useful as a "before" baseline). `checkpoints/` now contains only `_broken_2026-06-14_residual_bug/`, `_incompatible_pre_adaln/`, and `verify/`.
  - The training process (PID 26132) was stopped on 2026-06-14; no `wally.cli.train` python processes remain.
- **Hardware**: no change. The new loss has the same computational cost as the old one.
- **External**: no new dependencies. TheRock PyTorch / ROCm / Windows-native path is unchanged.
