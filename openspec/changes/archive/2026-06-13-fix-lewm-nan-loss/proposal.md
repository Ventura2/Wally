## Why

Training of the LeWorldModel has been producing NaN losses since the first resumed run, corrupting all 96/100 weight tensors in the latest checkpoint (`checkpoints/checkpoint_66000.pt`) within ~2000 steps of resume. Root cause: the SIGReg regularization was reimplemented as an MLP-critic mutual-information estimator (MINE-style) with a `tanh`-bounded output, which is a fundamentally different and unstable algorithm compared to the closed-form Epps-Pulley SIGReg in the original LeWM paper. Secondary causes compound the problem: missing NaN guards allow a single bad batch to poison the run for ~21k steps, and the LR scheduler is not restored on resume (wasting 500 warmup steps and confusing the learning trajectory). All three must be fixed together to make the next 100k-step run produce a usable model.

## What Changes

- **Replace MLP-critic SIGReg with closed-form Epps-Pulley SIGReg** (faithful to the LeWM paper): stateless `SIGReg(knots=17, num_proj=1024)` module that computes the Epps-Pulley statistic on random projections of the encoder embeddings. Removes the `SIGRegCritic` network, the separate critic optimizer, and the adversarial update loop in the trainer.
- **Add a NaN/Inf guard** in the training step: if the loss is non-finite, log a warning, zero gradients, and skip the optimizer step (the model keeps its current weights) instead of corrupting all parameters.
- **Save and restore the LR scheduler state** in checkpoints so resume correctly continues at the saved LR (fixes the silent warmup re-run).
- **Sanitize data inputs** in the trainer with `torch.nan_to_num(frames, 0.0)` and `torch.nan_to_num(actions, 0.0)` before moving to the device, mirroring the le-wm `train.py:32` pattern.
- **Move BatchNorm out of autocast** in `SimpleCNNEncoder` by casting inputs to fp32 inside the encoder when AMP is active (or by wrapping BN in a `@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)` decorator).
- **Add a smoke regression test** that runs 50 training steps on synthetic data and asserts the loss is finite and the checkpoint weights are free of NaN/Inf.

No public CLI/config surface changes. The `alpha`, `num_proj`, `knots` SIGReg config fields are new; the deprecated `critic` field is removed (this is internal to the Trainer constructor, not part of the YAML config).

## Capabilities

### New Capabilities
- `lewm-numerical-stability`: Numerical-stability contracts for the LeWM training loop — finite-loss guarantee under all input conditions, correct resume semantics, closed-form SIGReg with bounded gradient norms. Provides the regression test scaffolding and the stability invariants the rest of the training stack can rely on.

### Modified Capabilities
- `lewm-training-loop`: The SIGReg requirement changes from "MLP-critic MI estimator" to "closed-form Epps-Pulley statistic on random projections"; the critic training scenario is removed; a NaN-guard requirement and a scheduler-resume requirement are added.
- `minecraft-lewm-training`: The loss description is updated to reflect the closed-form SIGReg (stateless, non-negative) instead of the critic-based formulation.

## Impact

- **Code**: `src/wally/training/sigreg.py` (rewrite), `src/wally/training/losses.py` (update), `src/wally/training/trainer.py` (drop critic loop, add NaN guard, restore scheduler), `src/wally/training/checkpoint.py` (persist scheduler state), `src/wally/models/cnn_encoder.py` (BN fp32 in autocast), `src/wally/config/training.py` (add `num_proj`/`knots`), `configs/lewm_default.yaml` (add SIGReg params), `src/wally/cli/train.py` (thread new config fields).
- **Tests**: `tests/test_training_utils.py` (extend for scheduler round-trip), new `tests/test_lewm_numerical_stability.py` (synthetic 50-step smoke run, finite-loss + NaN-free checkpoint assertions).
- **Checkpoints**: existing checkpoints (`checkpoint_44000.pt` and earlier) remain loadable; the `critic_optimizer_state_dict` key is ignored on load. New checkpoints no longer include a critic state dict.
- **No user-visible CLI breaking changes** — only an additional config field with a sensible default.
