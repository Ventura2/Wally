# Trajectory Validator

## Purpose

Validates WebDataset shards for schema compliance, provides inspection commands, and reports dataset statistics.

## Requirements

### Requirement: Inspect shard contents
The system SHALL provide a CLI command to inspect the contents of one or more WebDataset shards and display summary statistics.

#### Scenario: Inspect single shard
- **WHEN** the user runs the inspect command on a shard file
- **THEN** the tool displays: transition count, episode count, observation shape, action keys, and timestamp range

#### Scenario: Inspect full dataset directory
- **WHEN** the user runs the inspect command on a shard directory
- **THEN** the tool reads the manifest and displays: total transitions, total episodes, shard count, and per-shard summaries

### Requirement: Validate shard schema compliance
The system SHALL validate that every sample in a shard conforms to the expected schema.

#### Scenario: Valid shard passes validation
- **WHEN** a shard contains only well-formed samples with `.jpg` and `.json` pairs
- **THEN** the validator reports zero errors and exits with code 0

#### Scenario: Missing JSON sidecar detected
- **WHEN** a shard contains a `.jpg` file without a corresponding `.json` file
- **THEN** the validator reports the missing sidecar with the sample key and exits with code 1

#### Scenario: Corrupt JPEG detected
- **WHEN** a shard contains a `.jpg` file that cannot be decoded
- **THEN** the validator reports the corrupt sample key and exits with code 1

### Requirement: Report action distribution
The system SHALL compute and display action distribution statistics across inspected shards.

#### Scenario: Action distribution displayed
- **WHEN** the user runs the inspect command with the `--actions` flag
- **THEN** the tool displays per-action-key statistics including mean, std, min, max for continuous values and value counts for discrete values

### Requirement: Validate minimum transition count
The system SHALL verify that a dataset meets the minimum transition count requirement.

#### Scenario: Dataset meets minimum
- **WHEN** the dataset contains at least 100,000 valid transitions
- **THEN** the validator reports "PASS: minimum transition count met"

#### Scenario: Dataset below minimum
- **WHEN** the dataset contains fewer than 100,000 valid transitions
- **THEN** the validator reports "FAIL: only N transitions found, minimum is 100,000" and exits with code 1

### Requirement: Sample visualization
The system SHALL support extracting and saving sample observations as image files for visual inspection.

#### Scenario: Extract sample frames
- **WHEN** the user runs the inspect command with `--samples N`
- **THEN** N random observations are decoded and saved as PNG files in a specified output directory
