## 1. Subgoal Detection

- [x] 1.1 Create `src/wally/planner/subgoal_detector.py` with `SubgoalDetector` class skeleton and `SubgoalDetectorConfig` (Pydantic model with threshold, smoothing_window, min_segment_length)
- [x] 1.2 Implement `compute_prediction_errors()` method that takes a trained LeWorldModel and trajectory (frames + actions), returns per-step L2 prediction errors in latent space
- [x] 1.3 Implement `smooth_errors()` method applying moving average filter with configurable window size
- [x] 1.4 Implement `detect_change_points()` method that finds local maxima in smoothed error signal exceeding threshold, with minimum segment length enforcement (merge adjacent segments by removing lower-magnitude point)
- [x] 1.5 Implement `extract_abstract_transitions()` method that segments trajectory at change points and produces (start_latent, end_latent, macro_action) tuples with mean-pooled action encoding
- [x] 1.6 Write unit tests for SubgoalDetector: prediction error computation, smoothing, change point detection on synthetic data, abstract transition extraction, config validation
- [x] 1.7 Add `SubgoalDetector` to `src/wally/planner/__init__.py` exports

## 2. High-Level World Model and Planner

- [x] 2.1 Create `src/wally/planner/high_level_planner.py` with `HighLevelPlanner` class and `HighLevelPlannerConfig` (macro_horizon: 5-10, population_size, n_iterations)
- [x] 2.2 Implement high-level LeWorldModel training support: accept abstract transition dataset, train separate predictor head with shared encoder weights, save checkpoint compatible with `LatentRollout`
- [x] 2.3 Implement `plan_subgoals()` method: encode current/goal frames, run CEM over macro actions through high-level world model, return sequence of subgoal latents
- [x] 2.4 Implement subgoal-to-low-level interface: convert subgoal latent sequence into per-subgoal targets for the low-level planner
- [x] 2.5 Implement sequential subgoal execution loop with timeout detection: advance to next subgoal when current is reached, flag failure on timeout
- [x] 2.6 Implement replanning on subgoal failure: request new subgoal sequence from high-level planner starting from current state
- [x] 2.7 Write unit tests for HighLevelPlanner: subgoal planning, sequential execution, timeout detection, replanning
- [x] 2.8 Add `HighLevelPlanner` to `src/wally/planner/__init__.py` exports

## 3. Gradient-Based MPC

- [x] 3.1 Create `src/wally/planner/gradient_mpc.py` with `GradientMPC` class and `GradientMPCConfig` (learning_rate, n_refinement_steps, grad_clip_norm, warm_start enabled)
- [x] 3.2 Implement differentiable rollout mode in `LatentRollout`: add `gradient_policy="straight_through"` support that maintains gradient flow through autoregressive prediction chain
- [x] 3.3 Implement `refine_actions()` method: take initial action sequence, perform gradient descent on cost function (latent distance to goal), clamp actions to bounds after each step, apply gradient clipping
- [x] 3.4 Implement warm-starting: accept optional policy network output as initial mean for CEM optimization in `GoalConditionedPlanner`
- [x] 3.5 Integrate gradient refinement as post-CEM stage: `GradientMPC.refine_actions()` called after `CEMOptimizer.optimize()` when enabled
- [x] 3.6 Write unit tests for GradientMPC: differentiable rollout gradient flow, action refinement reduces cost, action bounds enforcement, warm-start integration, config validation
- [x] 3.7 Add `GradientMPC` to `src/wally/planner/__init__.py` exports

## 4. Memory Augmentation

- [x] 4.1 Create `src/wally/models/recurrent_encoder.py` with `RecurrentEncoder` class wrapping ViTEncoder + single-layer LSTM, with configurable hidden_size and memory_length
- [x] 4.2 Implement `forward()` method: encode frame with ViT, mean-pool patch tokens, pass through LSTM with hidden state, return context-augmented latent
- [x] 4.3 Implement hidden state management: `reset_hidden()`, `get_hidden()`, `set_hidden()` methods; auto-reset on task completion
- [x] 4.4 Implement sequence processing: `forward_sequence()` method that processes T frames through LSTM, returns T latents and final hidden state
- [x] 4.5 Implement backward compatibility: `RecurrentEncoder` must work as drop-in for `ViTEncoder` in existing planner; support `recurrence=False` to bypass LSTM
- [x] 4.6 Write unit tests for RecurrentEncoder: single frame encoding, sequence processing, hidden state persistence/reset, drop-in compatibility, recurrence bypass
- [x] 4.7 Add `RecurrentEncoder` to `src/wally/models/__init__.py` exports

## 5. Curriculum Training

- [x] 5.1 Create `src/wally/training/curriculum.py` with `CurriculumTrainer` class and `CurriculumConfig` (stages: list[int], loss_threshold, patience, mix_shorter_sequences)
- [x] 5.2 Implement stage progression logic: track current stage, monitor validation loss, advance when loss < threshold for `patience` consecutive epochs
- [x] 5.3 Implement data slicing: filter/slice training data to match current horizon stage length, optionally mix in shorter sequences
- [x] 5.4 Implement shaped cost function: add subgoal proximity reward term to planning cost, with configurable shaping_weight blending
- [x] 5.5 Implement curriculum state persistence: save/load curriculum checkpoint (current_stage, epoch_count, best_val_loss)
- [x] 5.6 Write unit tests for CurriculumTrainer: stage progression, data slicing, shaped cost computation, checkpoint save/load, config validation
- [x] 5.7 Add `CurriculumTrainer` to `src/wally/training/__init__.py` exports

## 6. Curiosity-Driven Exploration

- [x] 6.1 Create `src/wally/training/curiosity.py` with `CuriosityModule` class and `CuriosityConfig` (forward_model_arch, reward_scale, update_frequency)
- [x] 6.2 Implement forward dynamics model: small MLP head mapping (current_latent, action) → predicted_next_latent, using shared ViT encoder latent space
- [x] 6.3 Implement `compute_intrinsic_reward()` method: L2 distance between predicted and actual next latent, scaled by reward_scale
- [x] 6.4 Implement forward model training loop: train on collected trajectory data with encoded latents, configurable update frequency relative to world model
- [x] 6.5 Implement exploration priority scoring: assign collection priority based on prediction error under current forward model
- [x] 6.6 Write unit tests for CuriosityModule: intrinsic reward computation, forward model training, reward scaling, config validation
- [x] 6.7 Add `CuriosityModule` to `src/wally/training/__init__.py` exports

## 7. Safety and Robustness

- [x] 7.1 Create `src/wally/training/ensemble.py` with `EnsembleWorldModel` class and `EnsembleConfig` (ensemble_size: 3-5, uncertainty_threshold)
- [x] 7.2 Implement ensemble training: shared encoder, N separate predictor heads with independent initialization, trained on same data
- [x] 7.3 Implement `predict_with_uncertainty()` method: run all ensemble members, return mean prediction and variance across members
- [x] 7.4 Implement per-step uncertainty tracking in rollout: return cumulative uncertainty (sum of per-step variances) alongside trajectory
- [x] 7.5 Implement safe plan selection: filter candidates by cumulative uncertainty threshold, prefer lowest uncertainty among similar-cost plans, flag low-confidence fallback
- [x] 7.6 Implement pluggable safety constraints: register custom constraint functions that evaluate and reject candidate trajectories
- [x] 7.7 Write unit tests for EnsembleWorldModel: ensemble training, uncertainty computation, safe plan selection, constraint checking, config validation
- [x] 7.8 Add `EnsembleWorldModel` to `src/wally/training/__init__.py` exports

## 8. Hierarchical Planner Integration

- [x] 8.1 Create `src/wally/planner/hierarchical_planner.py` with `HierarchicalPlanner` class that orchestrates high-level and low-level planners with subgoal execution loop
- [x] 8.2 Implement `plan()` method: encode current/goal frames, get subgoal sequence from high-level planner, execute each subgoal with low-level planner (with gradient MPC), handle replanning
- [x] 8.3 Extend `GoalConditionedPlanner` to accept subgoal latent targets (skip goal frame encoding), warm-start from policy network, return uncertainty estimates when ensemble is configured
- [x] 8.4 Implement `HierarchicalPlannerConfig` (Pydantic model composing CEMConfig, HighLevelPlannerConfig, GradientMPCConfig, with subgoal_timeout and max_replans)
- [x] 8.5 Write integration tests: full hierarchical planning on synthetic multi-phase trajectories, subgoal failure and replanning, gradient refinement integration
- [x] 8.6 Add `HierarchicalPlanner` to `src/wally/planner/__init__.py` exports

## 9. CLI and Configuration

- [x] 9.1 Create hierarchical planning CLI entry point (`wally-plan-hierarchical`) with arguments for checkpoint paths, goal frame, config overrides
- [x] 9.2 Add YAML config support for hierarchical planner: load `HierarchicalPlannerConfig` from YAML with all sub-configs
- [x] 9.3 Add curriculum training CLI entry point (`wally-train-curriculum`) with stage configuration
- [x] 9.4 Register new CLI entry points in `pyproject.toml`

## 10. End-to-End Testing

- [x] 10.1 Write end-to-end test: subgoal detection → abstract transition extraction → high-level model training → hierarchical planning on synthetic data
- [x] 10.2 Write end-to-end test: curriculum training with progressive horizon on synthetic trajectory data
- [x] 10.3 Write end-to-end test: ensemble training → uncertainty estimation → safe plan selection
- [x] 10.4 Verify full test suite passes: `uv run pytest`
- [x] 10.5 Verify lint and typecheck pass: `uv run ruff check .` and `uv run mypy`
