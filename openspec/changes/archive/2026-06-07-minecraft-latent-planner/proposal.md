## Why

The Wally project is training a LeWorldModel-style latent dynamics model on Minecraft trajectories. The trained model is only useful if it can drive behavior: given a current frame and a goal frame, we need to produce an action sequence that makes the agent reach the goal. Model Predictive Control with the Cross-Entropy Method (CEM) is the standard recipe for this — sample candidate action sequences, roll the world model forward in latent space, score by distance to the goal latent, refit the sampling distribution, and repeat. Without a planner, the world model cannot be used as a controller.

## What Changes

- Add a `mpc-cem-planner` capability that implements the Cross-Entropy Method optimizer over action sequences in continuous action space (with a discrete-to-continuous action adapter for MineStudio's discrete action vocabulary).
- Add a `latent-rollout` capability that exposes the trained `LeWorldModel` as a reusable latent-space simulator: given an initial latent and a sequence of action latents, return a trajectory of predicted latents.
- Add a `goal-conditioned-planning` capability that exposes the end-to-end `plan(current_frame, goal_frame) -> action_sequence` interface, composing the frozen ViT-Tiny encoder (from `lewm-model`), the latent rollout, and the CEM optimizer.
- Add a goal-encoding mechanism that uses the same frozen ViT-Tiny encoder to map the goal frame into the latent space used for CEM scoring.
- Add a planning configuration (YAML) covering CEM hyperparameters (population size, elite fraction, iterations, action bounds, plan horizon) and the cost function (latent distance metric, optional goal-progress shaping).
- Add a CLI entry point (`wally-plan`) that loads a trained checkpoint and runs planning against a MineStudio environment or an offline (current, goal) frame pair.
- Add a planning smoke test that verifies the planner returns valid action shapes, respects action bounds, and reduces cost across CEM iterations on a synthetic linear-Gaussian dynamics toy problem.

## Capabilities

### New Capabilities

- `mpc-cem-planner`: Cross-Entropy Method optimizer — population sampling, elite selection, distribution refit, and iterative refinement over bounded continuous action sequences. Lives at `wally/planner/cem.py`.
- `latent-rollout`: Reusable latent-space rollout built on top of the trained `LeWorldModel` Transformer predictor — given `(z_0, action_sequence)` returns a latent trajectory. Lives at `wally/planner/rollout.py`.
- `goal-conditioned-planning`: High-level planner that fuses the frozen encoder, the latent rollout, the goal encoder, and the CEM optimizer to expose `plan(current_frame, goal_frame) -> action_sequence`. Also owns the YAML config schema and the `wally-plan` CLI. Lives at `wally/planner/plan.py` and `wally/planner/cli.py`.

### Modified Capabilities

- `minecraft-latent-planner`: The existing high-level stub spec is expanded with implementation-level requirements (action-space handling, CEM hyperparameters, latent-cost formulation, dependency on a trained `LeWorldModel` checkpoint).

## Impact

- **Code**: New `wally/planner/` Python subpackage (`cem.py`, `rollout.py`, `plan.py`, `cli.py`, `config.py`). Reuses the encoder/predictor from the `lewm-model` capability introduced in the `lewm-training` change; the world model is consumed as a frozen module.
- **Dependencies**: `torch` (already required), `numpy`, `pyyaml` (already required), `einops` (already required), `minestudio` (already required) for the live-environment CLI path.
- **Data**: No new datasets. The planner consumes a trained checkpoint produced by the `lewm-training` change and (optionally) live MineStudio observations.
- **Downstream**: Unlocks `evaluation` (planning success-rate metrics can now be computed) and any goal-conditioned agent work that wants to use the planner as a policy.
- **Performance**: CEM planning is compute-heavy; defaults should be conservative (small population, few iterations) with a documented escape hatch for more expensive runs.
