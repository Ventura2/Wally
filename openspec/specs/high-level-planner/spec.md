# High-Level Planner

## Purpose

Implement a high-level planner that operates over abstract subgoal-to-subgoal transitions, planning sequences of latent subgoals toward a final goal and providing them as targets for the low-level planner.

## Requirements

### Requirement: High-level world model training
The system SHALL train a LeWorldModel on abstract transitions (subgoal-to-subgoal) using the same encoder architecture as the low-level model but with a separate predictor trained to predict the next subgoal latent given the current subgoal latent and macro action.

#### Scenario: Train on abstract transition dataset
- **WHEN** an abstract transition dataset is provided (from subgoal detection)
- **THEN** the system trains a high-level LeWorldModel that predicts next subgoal latents from current subgoal latent + macro action

#### Scenario: Shared encoder with separate predictor
- **WHEN** training the high-level model
- **THEN** the system uses the same ViT encoder weights as the low-level model but trains a separate causal Transformer predictor

#### Scenario: Checkpoint compatibility
- **WHEN** high-level model training completes
- **THEN** the system saves a checkpoint compatible with the existing LatentRollout loading mechanism

### Requirement: High-level CEM planning
The system SHALL plan sequences of latent subgoals toward a final goal using CEM optimization over the high-level world model, supporting planning horizons of 5-10 macro steps.

#### Scenario: Plan subgoal sequence to goal
- **WHEN** a current frame and goal frame are provided
- **THEN** the system returns a sequence of 5-10 intermediate subgoal latents that form a path from current to goal

#### Scenario: CEM optimization over macro actions
- **WHEN** planning at the high level
- **THEN** the system uses CEM to optimize a sequence of macro actions that minimize latent distance to the goal through the high-level world model

#### Scenario: Configurable macro-step horizon
- **WHEN** a planning horizon is specified (default 5, max 10)
- **THEN** the system plans exactly that many macro steps

### Requirement: Subgoal conditioning for low-level planner
The system SHALL convert high-level subgoal latents into goal frames or goal latents that can be consumed by the low-level planner as targets.

#### Scenario: Provide subgoal targets
- **WHEN** the high-level planner produces a subgoal sequence
- **THEN** the system provides each subgoal latent as the goal target for the low-level planner, one at a time

#### Scenario: Sequential subgoal execution
- **WHEN** executing a plan with multiple subgoals
- **THEN** the system executes subgoals in order, advancing to the next subgoal only after the current one is reached or times out

### Requirement: Replanning on subgoal failure
The system SHALL detect when the low-level planner fails to reach a subgoal within a configurable timeout and trigger replanning.

#### Scenario: Subgoal timeout detection
- **WHEN** the low-level planner exceeds the maximum number of steps for a subgoal without reaching it
- **THEN** the system flags the subgoal as failed

#### Scenario: Request new subgoal on failure
- **WHEN** a subgoal is flagged as failed
- **THEN** the system requests a new subgoal sequence from the high-level planner starting from the current state
