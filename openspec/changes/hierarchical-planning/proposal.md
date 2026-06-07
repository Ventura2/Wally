## Why

The current Wally system supports single-horizon CEM-based planning with a flat LeWorldModel, limiting it to short-horizon tasks (~16 steps). Long-horizon Minecraft goals (e.g., "craft diamond sword") require hierarchical decomposition into subgoals, multi-scale temporal planning, and memory to handle partial observability. Implementing the full hierarchical planning pipeline unlocks the ability to plan and execute complex multi-phase tasks.

## What Changes

- Add subgoal detection module using THICK/C-RSSM algorithms to identify context-change points in trajectories
- Train a high-level LeWorldModel on temporally abstract transitions between subgoals (5-10 macro steps)
- Add CEM-based high-level planner that generates sequences of latent subgoals toward a final goal
- Extend low-level planner with gradient-based MPC for differentiable action optimization
- Add warm-starting to CEM/gradient optimization using auxiliary value/policy network (Dreamer-style)
- Integrate high-level and low-level planners with subgoal-by-subgoal execution and replanning on failure
- Add LSTM/Transformer-XL recurrent module to the encoder for observation history tracking
- Implement curriculum training with progressive horizon increases (8 → 16 → 32 → full)
- Add shaped costs (pseudo-rewards) for subgoal progress during planning
- Implement curiosity-based exploration using prediction error as intrinsic reward
- Add safety-critic module with constraint checking, ensemble uncertainty estimation, and safe plan selection

## Capabilities

### New Capabilities
- `subgoal-detection`: Context-change point detection (THICK/C-RSSM) and abstract transition extraction from trajectories
- `high-level-planner`: High-level LeWorldModel trained on abstract transitions with CEM-based subgoal sequence planning
- `gradient-mpc`: Gradient-based MPC for low-level differentiable action optimization with warm-starting
- `memory-augmentation`: LSTM/Transformer-XL recurrent encoder for tracking observation history and handling partial observability
- `curriculum-training`: Progressive horizon training schedule with shaped costs for subgoal-directed planning
- `curiosity-exploration`: Intrinsic curiosity module using prediction error as exploration reward for diverse data collection
- `safety-critic`: Constraint checking, ensemble/dropout uncertainty estimation, and safe plan selection

### Modified Capabilities
- `minecraft-latent-planner`: Integrate with hierarchical system — support warm-starting from value/policy network, accept subgoal targets from high-level planner, and trigger replanning on subgoal failure

## Impact

- **Code**: New modules under `src/wally/planner/` (high-level planner, gradient MPC, safety-critic), new module under `src/wally/models/` (recurrent encoder), new module for subgoal detection, new training pipelines for high-level model and curriculum
- **APIs**: New CLI entry points for hierarchical planning; planner interface extended to accept subgoal targets and return uncertainty estimates
- **Dependencies**: Potential addition of Transformer-XL or LSTM dependencies (already available in PyTorch); may need ensemble utilities
- **Data**: Requires diverse multi-phase trajectories for training; new abstract transition dataset format
- **Training**: Multi-stage training pipeline (subgoal detection → high-level model → low-level enhancement → curriculum → exploration/safety); significantly increased compute requirements
- **Hardware**: AMD RX 6700 XT with ROCm — must ensure all new modules are ROCm-compatible
