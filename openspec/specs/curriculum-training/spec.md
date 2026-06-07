# Curriculum Training

## Purpose

Enable progressive horizon curriculum training for the world model, starting with short sequences and advancing to full horizon as the model improves, along with shaped cost functions for subgoal-directed planning.

## Requirements

### Requirement: Progressive horizon training
The system SHALL train world models on increasing sequence lengths following a curriculum schedule with stages: 8-step, 16-step, 32-step, and full horizon.

#### Scenario: Start with short sequences
- **WHEN** curriculum training begins
- **THEN** the system trains on 8-step sequences using only trajectory segments of that length

#### Scenario: Progress to longer sequences
- **WHEN** the model achieves the target validation loss threshold for the current horizon stage
- **THEN** the system advances to the next horizon stage (e.g., 8 -> 16 -> 32 -> full)

#### Scenario: Configurable stage thresholds
- **WHEN** custom loss thresholds and patience values are provided
- **THEN** the system uses those values to determine when to advance between stages

### Requirement: Shaped costs for subgoal-directed planning
The system SHALL support shaped cost functions that include intermediate pseudo-rewards for progress toward subgoals, not just final goal distance.

#### Scenario: Subgoal proximity reward
- **WHEN** planning toward a subgoal and the rollout passes near the subgoal latent
- **THEN** the cost function includes a bonus (reduced cost) for proximity to the subgoal at each step, not just the final step

#### Scenario: Configurable shaping weight
- **WHEN** a shaping weight is configured
- **THEN** the system blends the shaped cost with the base cost using that weight

### Requirement: Curriculum configuration
The system SHALL support configurable curriculum parameters including horizon stages, loss thresholds, patience epochs, and whether to mix shorter sequences at later stages.

#### Scenario: Custom curriculum schedule
- **WHEN** a custom list of horizon stages is provided (e.g., [4, 8, 16])
- **THEN** the system follows that exact schedule instead of the default

#### Scenario: Sequence mixing at later stages
- **WHEN** sequence mixing is enabled for a stage
- **THEN** the training data includes a proportion of shorter sequences alongside the current horizon length

### Requirement: Curriculum state persistence
The system SHALL support saving and resuming curriculum training state, including the current stage, epoch count, and best validation loss.

#### Scenario: Save curriculum checkpoint
- **WHEN** curriculum training is interrupted or a stage completes
- **THEN** the system saves the current curriculum state to a checkpoint file

#### Scenario: Resume from checkpoint
- **WHEN** training resumes from a curriculum checkpoint
- **THEN** the system continues from the saved stage and epoch count
