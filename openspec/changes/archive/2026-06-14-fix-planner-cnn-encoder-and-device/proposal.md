## Why

Three pre-existing bugs in the planner integration make it impossible to use the planner with the project's default training configuration (CNN encoder) and unusable on a CUDA device:

1. `LatentRollout._load_from_checkpoint` does not pass `encoder_type` to the `LeWorldModel` constructor, so every checkpoint saved with `encoder_type=cnn` (the default in `configs/lewm_default.yaml`) fails to load with a `state_dict` mismatch.
2. `LeWorldModelAdapter.encode` calls `tokens.mean(dim=1)` to pool ViT tokens, but the CNN encoder already returns a 2D `(B, D)` tensor — `mean(dim=1)` averages across the embedding dimension and produces `(B, 1)`, so even if (1) is fixed the planner still crashes on CNN checkpoints.
3. `CEMOptimizer.optimize` samples action candidates on CPU regardless of the world model's device, so any planner instantiation with `device="cuda"` raises `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same`.

These three bugs were discovered while writing `tools/eval_goals.py` (which works around them inline). Fixing them in the project removes the workarounds and unblocks real `wally-plan` / `wally-play` / `wally-plan-hierarchical` runs against the actual training checkpoints.

## What Changes

- **`LatentRollout.from_checkpoint`** (`src/wally/planner/rollout.py`): read `encoder_type` from the model config in the checkpoint and pass it to `LeWorldModel(...)` so CNN checkpoints load.
- **`LeWorldModelAdapter.encode`** (`src/wally/planner/rollout.py`): branch on the model's actual encoder type — for CNN, return the encoder's 2D output as-is; for ViT, mean-pool over the token axis. No more silent shape collapse.
- **CEM → world model device alignment**: `CEMOptimizer.optimize` shall accept a `device` argument (or otherwise ensure candidate actions live on the same device as the cost function's model tensors) so planners on `cuda` work without manual workarounds. Plumb the device through `GoalConditionedPlanner`, `HierarchicalPlanner`, `GradientMPC`, and any other call sites.
- **Add a test** that loads the existing `checkpoints/checkpoint_1000.pt` (CNN encoder) via `LatentRollout.from_checkpoint` and runs a small CEM rollout end-to-end on CPU. Optionally also a CPU↔CUDA test (skipped if CUDA is unavailable).
- **Remove the inline workarounds** in `tools/eval_goals.py` (`_load_world_model`, `_CNNCompatibleAdapter`, the `cpu` device default) so the eval tool relies on the now-fixed planner.

No public API removals. No new dependencies. No behaviour change for the existing ViT-encoder, CPU-only flow beyond making the bugs go away.

## Capabilities

### New Capabilities

- *None* — this is a bug fix that hardens existing capabilities.

### Modified Capabilities

- `latent-rollout`: add a requirement that `from_checkpoint` honours the `encoder_type` field stored in the training config (or, since the model config is currently not embedded in the checkpoint, accepts the model config as an explicit argument). Also add a requirement that `LeWorldModelAdapter.encode` returns a `(B, Z)` latent for both ViT and CNN encoders — no shape collapse on CNN.
- `mpc-cem-planner`: add a requirement that action samples produced by `CEMOptimizer.optimize` live on the same device as the tensors the cost function consumes, and that planners (`GoalConditionedPlanner`, `HierarchicalPlanner`, `GradientMPC`) thread the configured device through to the cost function so end-to-end device alignment is guaranteed.

## Impact

- Code: `src/wally/planner/rollout.py` (encoder_type + CNN encode + device), `src/wally/planner/cem.py` (device), `src/wally/planner/plan.py`, `src/wally/planner/hierarchical_planner.py`, `src/wally/planner/gradient_mpc.py`, `src/wally/planner/high_level_planner.py` (plumb device through to cost function).
- Tests: extend `tests/test_latent_rollout.py` with a CNN-encoder load + rollout test, and `tests/test_cem.py` (and any planner tests) with a device-alignment test.
- Tooling: `tools/eval_goals.py` (remove the three workarounds now that the bugs are fixed in the project).
- Public API: `CEMOptimizer.optimize` gains an optional `device` keyword. Default behaviour (CPU sampling) is preserved for backward compatibility.
- Risk: low. The fixes are targeted; the existing CPU + ViT path keeps working because the new branches replicate the old behaviour in that case.
