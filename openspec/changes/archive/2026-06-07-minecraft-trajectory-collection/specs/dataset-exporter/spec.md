## ADDED Requirements

### Requirement: Export trajectories as WebDataset shards
The system SHALL convert trajectory buffers into WebDataset-compatible `.tar` shards where each sample contains one transition.

#### Scenario: Shard creation from buffer
- **WHEN** a buffer of N transitions is passed to the exporter
- **THEN** a `.tar` file is written containing N samples, each with observation, action, timestamp, and metadata

#### Scenario: Shard key naming convention
- **WHEN** a transition is written to a shard
- **THEN** the sample key SHALL follow the pattern `{episode_id}_{step_index:06d}` with files `{key}.jpg` (observation), `{key}.json` (action + metadata)

### Requirement: JPEG observation encoding
The system SHALL encode each RGB observation as a JPEG image within the shard.

#### Scenario: JPEG encoding with configurable quality
- **WHEN** an observation is written to a shard
- **THEN** the image is JPEG-encoded at the configured quality level (default: 85) and stored as a `.jpg` file in the tar

### Requirement: JSON action and metadata encoding
The system SHALL encode action vectors and episode metadata as JSON within each shard sample.

#### Scenario: JSON sidecar per sample
- **WHEN** a transition is written to a shard
- **THEN** a `.json` file is created containing the action dict, timestamp, episode_id, step_index, frame_skip, and seed

### Requirement: Dataset manifest generation
The system SHALL generate a manifest file listing all shards with summary statistics.

#### Scenario: Manifest written after export session
- **WHEN** an export session completes
- **THEN** a `manifest.json` file is written containing: total transitions, total episodes, shard count, shard file list, and per-shard transition counts

### Requirement: Shard size targeting
The system SHALL target approximately 1000 transitions per shard and preserve episode boundaries within shards.

#### Scenario: Episode not split across shards
- **WHEN** an episode boundary occurs near a shard size limit
- **THEN** the episode's transitions remain in a single shard even if the shard slightly exceeds the target size

### Requirement: Output directory management
The system SHALL write shards to a configurable output directory and create it if it does not exist.

#### Scenario: Output directory auto-creation
- **WHEN** the exporter is initialized with a non-existent output path
- **THEN** the directory is created and shards are written into it
