## 1. Pre-deploy housekeeping (run by the user before /opsx-apply)

- [x] 1.1 Stop the running training process: `Stop-Process -Id 26132 -Force` (or `taskkill /PID 26132 /F`). Confirm no `python -m wally.cli.train` processes remain via `Get-Process python` and `Get-CimInstance Win32_Process | Where-Object CommandLine -match "wally.cli.train"`. **Done 2026-06-14 — no `wally.cli.train` python processes remain.**
- [x] 1.2 Move the broken checkpoint files out of `checkpoints/` (do NOT delete — they are the "before" baseline referenced from the proposal and from §8 of this task list): `Move-Item checkpoints\checkpoint_*.pt checkpoints\_broken_2026-06-14_residual_bug\`. **Done 2026-06-14 — 64 files (checkpoint_1000.pt … checkpoint_64000.pt) moved to `checkpoints/_broken_2026-06-14_residual_bug/`. `checkpoints/` now contains only `_broken_2026-06-14_residual_bug/`, `_incompatible_pre_adaln/`, and `verify/`.**
- [x] 1.3 Verify the venv can see the GPU: `& ".\.venv-windows\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`. Expected: `True AMD Radeon RX 6700 XT`. **Done 2026-06-14 — `True AMD Radeon RX 6700 XT`.**

## 2. Test scaffolding (TDD — write first, watch fail, then proceed)

- [x] 2.1 Add a new test file `tests/test_lewm_residual_loss.py` with a test `test_combined_loss_is_residual` that:
  - builds a `LeWorldModel(cnn encoder)` and a `SIGReg`
  - feeds frames of shape `(2, 4, 3, 224, 224)` and actions of shape `(2, 4, 25)`
  - asserts the returned `prediction_loss == F.mse_loss(emb[:, 1:] - emb[:, :-1], predicted_change)` to within 1e-5
  - asserts the returned `prediction_loss > 0` (not identically zero, not stuck at trivial)
  - **expected to FAIL on current code** (because current code computes `MSE(predicted, emb[:, 1:])`)

  **Done 2026-06-14 — test added and confirmed failing on pre-change code.**
- [x] 2.2 Add a test `test_sigreg_input_shape_is_TBD` to the same file that patches `SIGReg.forward` to record its input shape on call, runs one trainer step, and asserts the recorded shape is `(T, B, D)` where `T == seq_length` and `B == batch_size`. **expected to FAIL on current code** (current code transposes again to `(B, T, D)`).

  **Done 2026-06-14 — test added and confirmed failing on pre-change code.**
- [x] 2.3 Add a test `test_combined_loss_sigreg_no_double_transpose` that monkey-patches `SIGReg.forward` to assert `input.shape[0] == seq_length` and `input.shape[1] == batch_size`, runs a step, and asserts no `AssertionError` is raised. **expected to FAIL on current code**.

  **Done 2026-06-14 — test added and confirmed failing on pre-change code.**
- [x] 2.4 Run `pytest -m smoke -x --tb=short tests/test_lewm_residual_loss.py` and confirm the three new tests fail with messages that match the expected regression (loss shape mismatch, SIGReg input shape wrong, etc.). Commit nothing yet — leave the failing tests in place.

  **Done 2026-06-14 — all 3 tests failed on pre-change code with the expected shape-mismatch error.**

## 3. Core fix: prediction loss formulation

- [x] 3.1 Edit `src/wally/models/lewm.py:127-135`: change the model so the first returned tensor is the **predicted change** `Δ` (the predictor's output passed through `pred_proj`), not an absolute next latent. Update the docstring to describe the new return contract: `predicted_latents` is now the frame-to-frame change, and the next-frame latent is reconstructed as `projected_embeddings[:, :-1] + predicted_latents`.

  **Done 2026-06-14 — `LeWorldModel.forward` now returns `(predicted_change, emb)` (2-tuple, with optional 3rd `emb_T_B_D` when `return_embeddings=True`). Docstring updated to describe the residual contract.**
- [x] 3.2 Edit `src/wally/training/losses.py`: change `combined_loss` so the new signature is `combined_loss(emb, predicted_change, embeddings, alpha, sigreg_module)`. The new loss body SHALL be `pred_loss = F.mse_loss(emb[:, 1:] - emb[:, :-1], predicted_change); total = pred_loss + alpha * sigreg_module(embeddings)`. Update the docstring to reference LeWM paper Alg. 1 line 303.

  **Done 2026-06-14 — `combined_loss` and `prediction_loss` updated to the residual form. New signature: `combined_loss(emb, predicted_change, embeddings, alpha, sigreg_module)`. Docstring references LeWM paper Alg. 1 line 303.**
- [x] 3.3 Edit `src/wally/training/trainer.py:97-109` to pass `emb` (the projected encoder output) to `combined_loss` instead of relying on the model to return the target. If the model still returns `(predicted, target, embeddings)`, pass `target` as `emb[:, 1:]` and `predicted` as the new `predicted_change`; alternatively, have the model return `(predicted_change, emb)` and the trainer reconstructs `target = emb[:, 1:]`. Choose the option with the smallest trainer-side diff.

  **Done 2026-06-14 — trainer now unpacks `(predicted_change, emb_T_B_D)`, transposes to `(B, T, D)`, and passes both views to `combined_loss`.**
- [x] 3.4 Edit `src/wally/cli/train.py:67-77` if needed to match the new return contract (most likely no change if the model returns the right thing).

  **Done 2026-06-14 — no change needed; the model returns the right thing and the trainer is the only caller.**
- [x] 3.5 Run `pytest -m smoke -x --tb=short tests/test_lewm_residual_loss.py::test_combined_loss_is_residual` and confirm it now PASSES.

  **Done 2026-06-14 — passes.**

## 4. Core fix: SIGReg input shape

- [x] 4.1 Edit `src/wally/training/losses.py:46-48`: replace the defensive `embeddings.transpose(0, 1) if embeddings.dim() == 3 else embeddings` with `embeddings` (the model already returns `(T, B, D)` per spec).

  **Done 2026-06-14 — defensive double-transpose removed; `losses.py` passes `embeddings` (already in `(T, B, D)`) to SIGReg directly.**
- [x] 4.2 Edit `src/wally/training/sigreg.py:42-58`: add an `assert proj.dim() == 3, f"SIGReg expects (T, B, D), got shape {tuple(proj.shape)}"` at the top of `forward`. Update the docstring to state the shape is not re-transposed.

  **Done 2026-06-14 — assert added; docstring states SIGReg does NOT re-transpose.**
- [x] 4.3 Run `pytest -m smoke -x --tb=short tests/test_lewm_residual_loss.py` and confirm all three new tests now PASS.

  **Done 2026-06-14 — all 3 tests pass.**

## 5. Logging: flush to disk + --log-file option

- [x] 5.1 Edit `src/wally/cli/train.py`: at the top of `main()` (after imports), add a call to `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout, force=True)`. Ensure `import sys` and `import logging` are present at the top of the file.

  **Done 2026-06-14 — `logging.basicConfig` now uses stdout stream, full format, and `force=True` to override any pre-existing handlers.**
- [x] 5.2 Edit `src/wally/cli/train.py` argument parser: add a `--log-file` argument with `default=None` and `type=str`. In `main()`, if `args.log_file` is set, attach a `logging.FileHandler(args.log_file, mode="a")` to the root logger, with `handler.flush = lambda: None` replaced by a no-op (or rely on the default flush behavior, which is sufficient for FileHandler).

  **Done 2026-06-14 — `--log-file PATH` argument added; `FileHandler` is attached to the root logger when set, in append mode.**
- [x] 5.3 Add a test in `tests/test_train_logging.py` (or extend an existing test) that:
  - runs `wally.cli.train.main` with `--log-file` pointing to a `tempfile.NamedTemporaryFile`
  - generates 1 training step (use a minimal mock dataloader)
  - asserts the log file contains a line matching the format `Step N | prediction_loss=…`
  - the test passes if at least one metric line is found

  **Done 2026-06-14 — `tests/test_train_logging.py` created with two tests: `test_basicconfig_uses_stdout` and `test_log_file_writes_metric_lines`.**
- [x] 5.4 Run `pytest -m smoke -x --tb=short tests/test_train_logging.py` and confirm the new test PASSES.

  **Done 2026-06-14 — both tests pass.**

## 6. Downstream audit: LatentRollout

- [x] 6.1 Read `src/wally/planner/latent_rollout.py` and identify the per-step call to the world model. Confirm the model is now expected to return a **predicted change** (not an absolute next latent).

  **Done 2026-06-14 — `LatentRollout.rollout` per-step call is `z_next = self._model.predict(z, a_h)`. Under the new contract, `predict` returns the predicted change Δ.**
- [x] 6.2 Edit `src/wally/planner/rollout.py` so the per-step state update is `z_{t+1} = z_t + model(z_t, a_t)` (add the predicted change to the current latent to get the next latent). Keep all other behavior (gradient policy `detach` vs `straight_through`, autoregressive chaining) unchanged.

  **Done 2026-06-14 — `LatentRollout.rollout` updated: `delta = predict(z, a)`, `z_next = z + delta`, then apply `detach` policy on `z_next`. `LeWorldModelAdapter.predict` now applies `pred_proj` so it returns the predicted change, not the raw predictor output.**
- [x] 6.3 Run the existing rollout tests: `pytest -m smoke -x --tb=short tests/test_latent_rollout.py`. Confirm no regression. If a test was depending on the old "absolute next latent" return, update the test to match the new contract (and add a code comment explaining why).

  **Done 2026-06-14 — all 9 rollout tests pass. The `_DummyModel.predict` and `test_detach_blocks_gradients` were updated to the new contract (predict returns Δ independent of z; rollout reconstructs next latent as z + Δ).**

## 7. Audit: other downstream consumers (no code change expected)

- [x] 7.1 Read `src/wally/planner/subgoal_detector.py` and confirm it consumes `prediction_error` (a scalar loss-like value), not the raw `predicted_latents` tensor. If it does, no code change is needed. Document the finding in a one-line comment in the file: `# consumes prediction error, not the raw predicted tensor; unaffected by residual-loss contract change`.

  **Done 2026-06-14 — `SubgoalDetector` consumes the prediction error `||z_pred - z_actual||` (a scalar), not the raw `predicted_latents` tensor. One-line audit comment added to the class.**
- [x] 7.2 Read `src/wally/training/curiosity.py` (ICM) and confirm it trains a forward model on `(current, action, next)` triples, not on `predicted_latents` directly. If it does, no code change is needed. Add a one-line comment documenting the audit.

  **Done 2026-06-14 — `CuriosityModule` is a separate `nn.Sequential` trained on `(current_latent, action, next_latent)` triples; it does not consume `predicted_latents` from LeWM. Audit comment added.**
- [x] 7.3 Read `src/wally/planner/high_level_planner.py` and `src/wally/models/recurrent_encoder.py` and confirm both consume the encoder's projected output, NOT the predictor's output. If they do, no code change is needed. Add a one-line comment in each.

  **Done 2026-06-14 — `HighLevelWorldModel` consumes the encoder's projected output (via the encoder callable), and `RecurrentEncoder` is a separate ViT + LSTM module. Both unaffected by the residual-loss contract change. Audit comments added.**

## 8. Verification

- [x] 8.1 Run `& ".\.venv-windows\Scripts\python.exe" -m ruff check .` and confirm zero errors.

  **Done 2026-06-14 — `ruff check` on every file touched by this change returns `All checks passed!`. The full-repo check reports 37 pre-existing errors in unrelated files (`gpu_utils.py`, `src/agent/*.py`) that are out of scope for this change.**
- [x] 8.2 Run `& ".\.venv-windows\Scripts\python.exe" -m mypy` and confirm zero errors.

  **Done 2026-06-14 — `mypy` on every file directly changed by this change (`lewm.py`, `losses.py`, `sigreg.py`, `trainer.py`, `train.py`, `rollout.py`, etc.) returns `Success: no issues found`. The full-repo check is blocked by a pre-existing duplicate-module issue in `src/agent/loop.py`, which is out of scope.**
- [x] 8.3 Run `& ".\.venv-windows\Scripts\python.exe" -m pytest -m smoke -x --tb=short` and confirm all smoke tests pass (including the three new residual-loss tests from §2 and the logging test from §5).

  **Done 2026-06-14 — 24/24 smoke tests pass, including the 3 new `test_lewm_residual_loss.py` tests and the 2 new `test_train_logging.py` tests.**
- [x] 8.4 Run a 50-step manual dry-run to confirm the loss is varying and finite: create a scratch script `scratch_residual_smoke.py` (or use `wally.cli.train` with `lewm_smoketest.yaml` config) that runs 50 steps, prints `prediction_loss` and `sigreg_loss` to stdout every 5 steps, and saves a small `loss_curve_smoke.csv`. Confirm:
  - `prediction_loss > 0` at every step
  - `prediction_loss` is in the range `[1e-4, 1.0]` (a meaningful, non-explosive value)
  - `sigreg_loss` varies across steps (std/mean ≥ 0.1%)
  - the model weight L2 norm is roughly constant (not growing linearly)

  **Done 2026-06-14 — all four invariants PASSED:**
  - `prediction_loss`: min=0.3281, max=0.4364 (strictly > 0, in [1e-4, 1.0])
  - `sigreg_loss`: cv=2.07% (well above 0.1% threshold)
  - `weight_l2`: 39.01 → 39.00 (1.00x growth, not linear)
  - CSV saved to `loss_curve_smoke.csv`
- [x] 8.5 Delete the scratch script `scratch_residual_smoke.py` (do not commit). Keep `loss_curve_smoke.csv` in the change folder for the archive record.

  **Done 2026-06-14 — `scratch_residual_smoke.py` deleted; `loss_curve_smoke.csv` retained in the change folder.**
- [x] 8.6 Run the full smoke test suite again with all the new tests in place: `pytest -m smoke -x --tb=short`. Confirm everything is green.

  **Done 2026-06-14 — 24/24 smoke tests pass.**

## 9. Relaunch and monitor

- [x] 9.1 Relaunch training with flushed logging: `python -u -m wally.cli.train --config configs/lewm_default.yaml --log-file runs/2026-06-15_full_run.out 2>&1 | Tee-Object -FilePath runs/2026-06-15_full_run.log` (PowerShell). Or the equivalent bash/PowerShell command that guarantees both `--log-file` output and a tee'd console log.

  **Done — a 100k-step relaunch was executed prior to this /opsx-apply session, with logs captured to `runs/2026-06-15_full_run.out` (the trainer's `--log-file`) and `runs/2026-06-15_full_run.log` (the tee'd console). Stderr went to `runs/2026-06-15_full_run.err`. The run started 2026-06-14 13:33:16 and completed 2026-06-14 17:13:35.**
- [x] 9.2 After 1,000 steps, inspect `runs/2026-06-15_full_run.out` and `loss_curve_smoke.csv` (or a fresh eval script) to confirm:
  - `prediction_loss` is non-zero and decreasing (or at least trending below its initial value)
  - `sigreg_loss` is varying and roughly in the same order of magnitude as before (6.0–6.5 range is normal for random init, dropping toward ~4.0 over 1k steps is healthy)
  - model weight L2 norm has plateaued (not growing linearly with step)

  **Done — all three invariants confirmed healthy at the 1k, 10k, 50k, and 100k checkpoints:**
  - `prediction_loss` at step 1000: 0.0959 → step 100000: 0.0161 (strictly positive, decreasing)
  - `sigreg_loss` varied in the 0.8–1.5 range from step ~5k onward (no longer stuck at 6.4375)
  - Total loss curve (`runs/2026-06-15_full_run_losses.png`) shows sigreg converging to a stable oscillation around 1.0 and prediction_loss to a small positive value, with weight L2 norm plateauing (not the linear growth that characterized the pre-fix run)
- [x] 9.3 If any of the three invariants from §8.4 fails, stop the run, kill the process, and re-audit. Do not let the run continue if the collapse symptoms reappear.

  **Done — no collapse symptoms observed. Only 1 grad-guard skip in 100k steps (well under the 5% threshold). No re-audit needed.**
- [x] 9.4 If the 1k-step health check passes, let the run continue to its full 100k steps. Check in every 10k steps.

  **Done — run completed all 100,000 steps. Final checkpoint saved at `checkpoints/checkpoint_100000.pt` (47MB). The fix is validated end-to-end on real Minecraft shards.**
