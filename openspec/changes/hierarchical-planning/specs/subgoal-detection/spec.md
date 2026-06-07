## ADDED Requirements

### Requirement: Prediction error computation
The system SHALL compute per-step prediction error from a trained LeWorldModel by comparing predicted latents against actual encoded latents for each frame in a trajectory.

#### Scenario: Compute prediction error for a trajectory
- **WHEN** a trajectory of T frames and T-1 actions is provided
- **THEN** the system returns a sequence of T-1 scalar prediction errors (L2 distance in latent space)

#### Scenario: Handle batched trajectories
- **WHEN** a batch of trajectories is provided
- **THEN** the system computes prediction errors independently for each trajectory in the batch

### Requirement: Context-change point detection
The system SHALL detect context-change points in a trajectory by identifying local maxima in the smoothed prediction error signal that exceed a configurable threshold.

#### Scenario: Detect phase transitions
- **WHEN** a trajectory contains distinct phases (e.g., mining then crafting)
- **THEN** the system identifies the frame indices where context changes occur

#### Scenario: Smoothing with moving average
- **WHEN** raw prediction errors are noisy
- **THEN** the system applies a moving average filter with configurable window size before detecting peaks

#### Scenario: Minimum segment length enforcement
- **WHEN** detected change points would create segments shorter than the configured minimum length
- **THEN** the system merges adjacent segments by removing the lower-magnitude change point

### Requirement: Abstract transition extraction
The system SHALL extract abstract transitions between consecutive context-change points, producing a dataset of (start_latent, end_latent, macro_action) tuples suitable for training a high-level world model.

#### Scenario: Extract transitions from segmented trajectory
- **WHEN** a trajectory is segmented at N context-change points
- **THEN** the system produces N-1 abstract transitions, each containing the start latent, end latent, and a summary of actions taken in that segment

#### Scenario: Macro action encoding
- **WHEN** an abstract transition spans multiple primitive actions
- **THEN** the system encodes the action sequence as a single macro action vector (mean-pooled or learned embedding)

### Requirement: Subgoal detection configuration
The system SHALL support configurable detection parameters including prediction error threshold, smoothing window size, and minimum segment length.

#### Scenario: Custom threshold configuration
- **WHEN** a user provides a custom threshold value
- **THEN** the system uses that threshold for context-change detection instead of the default

#### Scenario: Configuration validation
- **WHEN** invalid parameters are provided (e.g., negative threshold, window size larger than trajectory)
- **THEN** the system raises a validation error with a descriptive message
