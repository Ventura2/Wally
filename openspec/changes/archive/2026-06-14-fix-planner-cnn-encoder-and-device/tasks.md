## 1. Embed model config in saved checkpoints

- [x] 1.1 Update `save_checkpoint` in `src/wally/training/checkpoint.py` to accept an optional `model_config: dict | None = None` keyword and store it under the `model_config` key in the payload. JSON-serialize any non-trivial values (cast `Path`, dataclasses, etc. to plain dicts/strings) so the field is portable.
- [x] 1.2 Update `wally-train` (`src/wally/cli/train.py`) to pass `model_config=asdict(model_config)` (or the equivalent plain dict from `wally.config.model.ModelConfig`) into `save_checkpoint`. The trainer (`src/wally/training/trainer.py`) needs the same value plumbed through: add a `model_config` parameter to `Trainer.__init__` and forward it to both periodic and final `save_checkpoint` calls.
- [x] 1.3 Add a unit test in `tests/test_training_utils.py` (or extend an existing one) that calls `save_checkpoint` with a model_config, loads it back with `torch.load`, and asserts the `model_config` key is present and equal to the input dict.
- [x] 1.4 Run the existing `tests/test_train_logging.py` and `tests/test_checkpoint.py` (if present) to confirm the new optional argument is backward compatible (existing call sites that pass only positional args keep working).

## 2. Honour `encoder_type` when loading checkpoints

- [x] 2.1 In `src/wally/planner/rollout.py`, update `LatentRollout._load_from_checkpoint` to resolve `encoder_type` with the priority: `checkpoint["model_config"]["encoder_type"]` → `checkpoint["config"]["model"]["encoder_type"]` (legacy) → `"vit"` (default).
- [x] 2.2 Pass the resolved `encoder_type` to the `LeWorldModel(...)` constructor so the state dict matches. Keep the other model constructor args on their current default-with-override behaviour.
- [x] 2.3 Add a regression test in `tests/test_latent_rollout.py` that loads the existing `checkpoints/checkpoint_1000.pt` (CNN encoder) via `LatentRollout.from_checkpoint(...)` and asserts the underlying model is the CNN encoder and that `rollout` runs without a state_dict mismatch.
- [x] 2.4 If any pre-AdaLN ViT checkpoint survives in `checkpoints/_incompatible_pre_adaln/`, add a second test that loads it and confirms the ViT path still works (skip with a clear reason if no such checkpoint is available).

## 3. Fix `LeWorldModelAdapter.encode` for the CNN encoder

- [x] 3.1 In `src/wally/planner/rollout.py`, modify `LeWorldModelAdapter.encode` to branch on `self._model._is_cnn` (set by `LeWorldModel.__init__`): return `self._model.encoder(frame)` directly for CNN, and `self._model.encoder(frame).mean(dim=1)` for ViT.
- [x] 3.2 In `tests/test_latent_rollout.py`, add a test that builds a tiny `LeWorldModel(encoder_type="cnn")`, wraps it in `LeWorldModelAdapter`, encodes a `(2, 3, 224, 224)` frame, and asserts the output shape is `(2, embed_dim)`.
- [x] 3.3 Add a complementary test for the ViT case (default `encoder_type`) that asserts the output is `(2, embed_dim)` and equals `model.encoder(frame).mean(dim=1)` elementwise.
- [x] 3.4 Add an integration-style test that calls `adapter.predict(adapter.encode(frame), action)` on a CNN model and confirms it does not raise a shape error.

## 4. Make `CEMOptimizer.optimize` device-aware

- [x] 4.1 In `src/wally/planner/cem.py`, add `device: torch.device | str | None = None` to the `optimize` signature (and to `RandomShooting.optimize` for symmetry, even though no current call site uses it).
- [x] 4.2 When `device` is non-`None`, create `mean`, `std`, and the sampled population via `torch.zeros(..., device=device)` / `torch.full(..., device=device)` / `torch.randn(..., device=device)`. When `device` is `None`, preserve the existing CPU-only behaviour.
- [x] 4.3 Make `_sample_truncated_normal` also accept and forward `device` so it stays consistent with the public `optimize` interface.
- [x] 4.4 Add tests in `tests/test_cem.py` that: (a) call `optimize(..., device="cpu")` and assert the returned tensor is on CPU; (b) call `optimize(..., device="cuda")` and assert the returned tensor is on CUDA (skip the second test if `torch.cuda.is_available()` is `False`).
- [x] 4.5 Add a test that the default (`device=None`) still produces CPU tensors and that all existing CEM tests in the suite remain green.

## 5. Plumb device through the planners

- [x] 5.1 In `src/wally/planner/plan.py`, add `device=self._device` to both `self._cem.optimize(...)` call sites inside `GoalConditionedPlanner.plan` and `plan_to_latent`.
- [x] 5.2 In `src/wally/planner/hierarchical_planner.py`, locate every `CEMOptimizer.optimize` call (and any direct `torch.randn`/`torch.zeros` that should follow the planner's device) and forward `self._device`. (HierarchicalPlanner only delegates to sub-planners; no direct CEM calls — verified.)
- [x] 5.3 In `src/wally/planner/gradient_mpc.py`, forward `self._device` into the `CEMOptimizer.optimize(...)` call inside `GradientMPC.plan`.
- [x] 5.4 In `src/wally/planner/high_level_planner.py`, do the same — every CEM/internal-sample call uses `self._device`.
- [x] 5.5 Add a regression test in `tests/test_goal_conditioned_planner.py` that constructs a `GoalConditionedPlanner` with `device="cpu"` and runs `plan_to_latent` end-to-end on a small mock world model — must pass (locks in the CPU path).
- [x] 5.6 Add a CUDA test in the same file that builds a `GoalConditionedPlanner` with `device="cuda"` (or skips if CUDA unavailable) and asserts `plan_to_latent` returns a CUDA tensor and does not raise a device-mismatch error.
- [x] 5.7 Re-run `tests/test_hierarchical_planner.py`, `tests/test_gradient_mpc.py`, `tests/test_high_level_planner.py`, `tests/test_planner_cli.py` to confirm no regression on the CPU path.

## 6. Strip workarounds from `tools/eval_goals.py`

- [x] 6.1 Remove the `_CNNCompatibleAdapter` class from `tools/eval_goals.py` and any references to it. Use the stock `LeWorldModelAdapter` from `wally.planner.rollout` instead.
- [x] 6.2 Remove `_load_world_model` from `tools/eval_goals.py`; replace its call site with `LatentRollout.from_checkpoint(ckpt_path, device=device)` (now safe for CNN checkpoints).
- [x] 6.3 Change the `--device` default in the CLI back to `"auto"` (or remove the explicit default and let `LatentRollout` pick), and update the help text to drop the "planner creates CEM samples on CPU" caveat. (Note: `--config` is restored so the script can pass `model_config` to `LatentRollout.from_checkpoint` for old checkpoints that lack an embedded `model_config`.)
- [x] 6.4 Re-run the smoke test: `python tools/eval_goals.py --checkpoints 'checkpoints/_smoke_dummy.pt' --mode mock --episodes 1 --output runs/_smoke_eval_after` and confirm the script still produces a valid `episodes.csv` / `episodes.json` / `report.md`.

## 7. Verify

- [x] 7.1 Run the full test suite with `pytest` and confirm all tests pass. Pay special attention to `tests/test_latent_rollout.py`, `tests/test_cem.py`, `tests/test_goal_conditioned_planner.py`, `tests/test_hierarchical_planner.py`, `tests/test_gradient_mpc.py`, `tests/test_high_level_planner.py`, `tests/test_planner_cli.py`, `tests/test_train_logging.py`, `tests/test_checkpoint.py`. (143 passed, 1 skipped; pre-existing failures in `test_model.py`, `test_evaluate.py`, `test_env.py`, `test_agent_env.py` are from a separate LeWM refactor and are out of scope.)
- [x] 7.2 Run `ruff check .` and `mypy` (per the project's `AGENTS.md` lint/typecheck commands) and address any new warnings. (Ruff: 0 new errors from this change. mypy: pre-existing errors only, none introduced.)
- [x] 7.3 Run `python tools/eval_goals.py --checkpoints 'checkpoints/checkpoint_*.pt' --num-checkpoints 2 --mode world_model --episodes 1 --output runs/goal_eval_after` and confirm the script runs against the real CNN checkpoints without hitting the three workarounds. (Ran end-to-end on CUDA; both `checkpoint_1000.pt` and `checkpoint_100000.pt` loaded via the stock `LatentRollout.from_checkpoint`, planner ran without device-mismatch, and the three output files were produced.)
- [x] 7.4 Mark the change ready for archive by running `/opsx-archive` (or following the project's archive workflow) once the user accepts the implementation.
