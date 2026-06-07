# Trajectory Collector

## Purpose

Collects Minecraft gameplay trajectories by connecting to a MineStudio environment, capturing RGB observations and agent actions, and buffering transitions for export.

## Requirements

### Requirement: Connect to Minecraft via MineStudio
The system SHALL initialize a MineStudio environment and maintain a stable connection to a running Minecraft instance throughout a collection session.

#### Scenario: Successful environment initialization
- **WHEN** the collector is started with valid MineStudio configuration
- **THEN** a MineStudio environment is created and the first observation is returned within 30 seconds

#### Scenario: Connection loss handling
- **WHEN** the MineStudio connection is lost during collection
- **THEN** the collector SHALL flush the current buffer to disk and raise a descriptive error

### Requirement: Capture RGB observations
The system SHALL capture RGB pixel observations from the player's first-person perspective at each environment step.

#### Scenario: Observation capture per step
- **WHEN** the environment executes a step
- **THEN** the current RGB frame is captured and added to the trajectory buffer

#### Scenario: Observation resize to 224x224
- **WHEN** an observation is captured
- **THEN** the image SHALL be resized to exactly 224x224 pixels using bilinear interpolation before storage

### Requirement: Record agent actions
The system SHALL record the action vector executed at each environment step alongside the corresponding observation.

#### Scenario: Action recorded with observation
- **WHEN** a step is executed with a given action
- **THEN** the action is stored as a JSON-serializable dictionary paired with the observation taken after the action

### Requirement: Configurable frame skip
The system SHALL support a configurable frame skip parameter that determines how many environment steps are executed between recorded transitions.

#### Scenario: Frame skip of 4
- **WHEN** frame_skip is set to 4
- **THEN** the environment executes 4 steps per recorded transition, and only the final observation and cumulative reward are stored

#### Scenario: Frame skip of 1
- **WHEN** frame_skip is set to 1
- **THEN** every environment step produces a recorded transition with no frames skipped

### Requirement: Timestamp each transition
The system SHALL attach a Unix timestamp (millisecond precision) to each recorded transition.

#### Scenario: Timestamp present
- **WHEN** a transition is recorded
- **THEN** the transition contains a `timestamp` field with the current Unix time in milliseconds

### Requirement: Episode metadata tracking
The system SHALL track episode boundaries and attach episode metadata to each transition.

#### Scenario: Episode start metadata
- **WHEN** a new episode begins (environment reset)
- **THEN** all transitions in that episode share a unique `episode_id` and include the episode `seed` in metadata

#### Scenario: Episode end detection
- **WHEN** the environment signals `done=True`
- **THEN** the current episode is marked complete and a new episode begins on the next step

### Requirement: Trajectory buffer management
The system SHALL maintain an in-memory buffer of transitions and flush to disk when a configurable size threshold is reached.

#### Scenario: Buffer flush on threshold
- **WHEN** the buffer reaches the configured max size (default: 1000 transitions)
- **THEN** the buffer contents are passed to the exporter and the buffer is cleared

#### Scenario: Buffer flush on shutdown
- **WHEN** the collector is stopped gracefully
- **THEN** any remaining transitions in the buffer are flushed before exit
