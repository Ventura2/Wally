## 1. Replace SIGReg algorithm

- [x] 1.1 Rewrite `src/wally/training/sigreg.py`: drop `SIGRegCritic` and `sigreg_loss`; add a parameterless `SIGReg(knots=17, num_proj=1024)` module that computes the closed-form Epps-Pulley statistic on `(T, B, D)` embeddings.
- [x] 1.2 Update `src/wally/training/losses.py`: change `combined_loss` signature to take `(predicted, target, embeddings, alpha)`, drop the `critic` argument, and return `pred_loss + alpha * sigreg_loss(embeddings)`.
- [x] 1.3 Update `src/wally/config/training.py` to add `sigreg_num_proj` and `sigreg_knots` fields with defaults 1024 and 17; remove the now-unused critic fields.
- [x] 1.4 Update `src/wally/cli/train.py` to instantiate the new `SIGReg` module and pass it (instead of a critic) to the Trainer constructor.
- [x] 1.5 Update `src/wally/training/trainer.py`: drop `self.critic` / `self.critic_optimizer`; instantiate `SIGReg` instead; pass the encoder embeddings into `combined_loss`.

## 2. Expose encoder embeddings to the loss

- [x] 2.1 Update `src/wally/models/lewm.py` so the model can return the per-frame encoder embeddings as a third output, gated by a `return_embeddings: bool = False` flag (kept off by default to avoid breaking inference callers like planning/rollout).

## 3. Harden the training loop

- [x] 3.1 In `src/wally/training/trainer.py:_training_step`, after computing the loss, check `torch.isfinite(total_loss).all()`. If false, log a warning with the step number, call `self.optimizer.zero_grad()`, advance `global_step`, and return the previous-step metrics without calling `optimizer.step()`.
- [x] 3.2 In `src/wally/training/trainer.py:_training_step`, apply `torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)` to `frames` and `actions` immediately after the `.to(self.device)` call.

## 4. Persist and restore scheduler state

- [x] 4.1 In `src/wally/training/checkpoint.py`, add an optional `scheduler` argument to `save_checkpoint`; persist `scheduler.state_dict()` as `"scheduler_state_dict"` in the checkpoint dict.
- [x] 4.2 In `src/wally/training/checkpoint.py`, add an optional `scheduler` argument to `load_checkpoint`; if the key is present, call `scheduler.load_state_dict(...)`; if absent and the scheduler is provided, set `scheduler.last_epoch = global_step - 1` and log an info message.
- [x] 4.3 In `src/wally/training/trainer.py`, pass `self.scheduler` to `save_checkpoint` and `load_checkpoint`; gracefully tolerate legacy checkpoints that lack `critic_optimizer_state_dict`.

## 5. Move BatchNorm out of autocast

- [x] 5.1 In `src/wally/models/cnn_encoder.py`, wrap `forward` with `@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)` so BatchNorm runs in fp32 while convs use autocast.

## 6. Update default config

- [x] 6.1 In `configs/lewm_default.yaml`, change `alpha: 0.01` (was 0.1) to match the paper's published weight for the closed-form SIGReg.

## 7. Tests

- [x] 7.1 In `tests/test_training_utils.py`, add `TestCheckpoint` cases for: scheduler round-trip preserves LR; legacy checkpoint (no `scheduler_state_dict`, no `critic_optimizer_state_dict`) loads without raising; loaded model has identical weights.
- [x] 7.2 Create `tests/test_lewm_numerical_stability.py` with: (a) `test_sigreg_finite_on_degenerate` — SIGReg on zeros and on Gaussian embeddings returns finite, non-negative values; (b) `test_smoke_run_finite_loss` — 50-step trainer run on synthetic data with default config logs all finite losses; (c) `test_smoke_run_checkpoint_no_nan` — checkpoint after 50 steps has no NaN/Inf tensors; (d) `test_nan_guard_skips_step` — patched NaN loss for one batch is recovered, model weights unchanged for that step, next batch finite; (e) `test_resume_does_not_rewarmup` — save at step 10000, reload, verify LR is not near zero.

## 8. Verify

- [x] 8.1 Run `.\.venv-windows\Scripts\python.exe -m pytest -m smoke -x --tb=short` — all tests pass.
- [x] 8.2 Run `.\.venv-windows\Scripts\python.exe -m ruff check .` — clean.
- [x] 8.3 Run `.\.venv-windows\Scripts\python.exe -m mypy` — clean.
- [x] 8.4 Manually inspect a 100-step dry run of `wally-train` on a 1-batch synthetic dataset to confirm SIGReg loss is `O(1)` and finite.
