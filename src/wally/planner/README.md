# Planner

Goal-conditioned planning in latent space using Cross-Entropy Method (CEM)
with a trained LeWorldModel. Supports both flat MPC and hierarchical planning
with subgoal decomposition.

## Public API

### CEMConfig

Configuration dataclass for the CEM optimizer.

```python
from wally.planner.config import CEMConfig

config = CEMConfig(
    population_size=64,
    elite_frac=0.1,
    n_iterations=5,
    horizon=8,
    action_low=-1.0,
    action_high=1.0,
    gradient_policy="detach",
)
```

Load from YAML:

```python
config = CEMConfig.from_yaml("configs/planner/default.yaml")
```

| Field             | Type                              | Default   | Description                                      |
|-------------------|-----------------------------------|-----------|--------------------------------------------------|
| population_size   | int (>1)                          | 64        | Number of action sequences sampled per iteration |
| elite_frac        | float in (0, 1)                   | 0.1       | Fraction of population kept as elites            |
| n_iterations      | int (>=1)                         | 5         | CEM refinement iterations                        |
| horizon           | int (>=1)                         | 8         | Action sequence length                           |
| action_low        | float                             | -1.0      | Lower clamp for actions                          |
| action_high       | float                             | 1.0       | Upper clamp for actions                          |
| gradient_policy   | "detach" \| "straight_through"    | "detach"  | How gradients flow through latent rollouts       |

### CEMOptimizer

Cross-Entropy Method optimizer. Samples action sequences from a truncated
normal distribution, evaluates them with a cost function, and iteratively
refits the distribution around the elite subset.

```python
from wally.planner.cem import CEMOptimizer

cem = CEMOptimizer()
actions, cost_history = cem.optimize(
    cost_fn,
    horizon=8,
    action_dim=25,
    population_size=64,
    elite_frac=0.1,
    n_iterations=5,
)
```

### LatentRollout

Wraps a trained LeWorldModel to unroll action sequences in latent space.

```python
from wally.planner.rollout import LatentRollout

rollout = LatentRollout.from_checkpoint("checkpoints/model.pt", device="cuda")
latents = rollout.rollout(z_0, actions)
```

Constructor arguments:
- `model` -- a `WorldModelProtocol` instance (optional if `checkpoint_path` given)
- `checkpoint_path` -- path to a `.pt` checkpoint (optional if `model` given)
- `device` -- torch device string or object
- `gradient_policy` -- `"detach"` or `"straight_through"`

### GoalConditionedPlanner

High-level planner that combines CEM optimization with latent rollouts to
find action sequences that reach a goal state.

```python
from wally.planner.plan import GoalConditionedPlanner

planner = GoalConditionedPlanner(
    world_model=rollout,
    encoder=rollout._model.encode,
    config=config,
    device="cuda",
)
actions = planner.plan(current_frame, goal_frame)
```

Constructor arguments:
- `world_model` -- a `LatentRollout` instance
- `encoder` -- callable `(B,C,H,W) -> (B,Z)` that encodes frames to latents
- `config` -- a `CEMConfig` instance
- `device` -- torch device (auto-detects CUDA if omitted)
- `cost_fn` -- optional custom cost `(z_H, z_g) -> (B,)`; defaults to L2
- `action_dim` -- action dimensionality (default 25)

## End-to-end example

```python
from wally.planner.config import CEMConfig
from wally.planner.rollout import LatentRollout
from wally.planner.plan import GoalConditionedPlanner

config = CEMConfig.from_yaml("configs/planner/default.yaml")
rollout = LatentRollout.from_checkpoint("checkpoints/model.pt", device="cuda")
planner = GoalConditionedPlanner(
    rollout,
    rollout._model.encode,
    config,
    device="cuda",
)

actions = planner.plan(current_frame, goal_frame)
```

## Action vocabulary

The `actions` module provides a MineStudio action vocabulary and conversion
utilities between continuous and discrete action representations.

```python
from wally.planner.actions import MineStudioActionVocab, continuous_to_discrete

vocab = MineStudioActionVocab.default()
discrete = continuous_to_discrete(actions, vocab)
```

## Hierarchical Planning

For long-horizon tasks, the hierarchical planner decomposes goals into subgoals
and executes them sequentially with replanning on failure.

### SubgoalDetector

Detects context-change points in trajectories using prediction error analysis:

```python
from wally.planner.subgoal_detector import SubgoalDetector, SubgoalDetectorConfig

config = SubgoalDetectorConfig(threshold=1.0, smoothing_window=5, min_segment_length=8)
detector = SubgoalDetector(config)

errors = detector.compute_prediction_errors(model, frames, actions)
smoothed = detector.smooth_errors(errors)
change_points = detector.detect_change_points(smoothed)
transitions = detector.extract_abstract_transitions(latents, actions, change_points)
```

### HighLevelPlanner

Plans sequences of latent subgoals using CEM over a high-level world model:

```python
from wally.planner.high_level_planner import HighLevelPlanner, HighLevelPlannerConfig

config = HighLevelPlannerConfig(macro_horizon=5, population_size=32)
planner = HighLevelPlanner(high_level_model, encoder, config)

subgoal_latents, cost = planner.plan_subgoals(current_frame, goal_frame)
targets = planner.subgoals_to_targets(subgoal_latents)
```

### GradientMPC

Gradient-based refinement of CEM-optimized action sequences:

```python
from wally.planner.gradient_mpc import GradientMPC, GradientMPCConfig

config = GradientMPCConfig(learning_rate=0.01, n_refinement_steps=10)
mpc = GradientMPC(world_model, encoder, config)

actions = mpc.plan(current_frame, goal_frame)
```

### HierarchicalPlanner

Orchestrates high-level and low-level planners with subgoal execution:

```python
from wally.planner.hierarchical_planner import HierarchicalPlanner, HierarchicalPlannerConfig

config = HierarchicalPlannerConfig(subgoal_timeout=50, max_replans=3)
planner = HierarchicalPlanner(high_level_planner, low_level_planner, config)

result = planner.plan(current_frame, goal_frame)
# result.actions, result.success, result.subgoals, result.replan_count
```
