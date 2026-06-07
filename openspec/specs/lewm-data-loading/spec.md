# lewm-data-loading

## Purpose

Data loading pipeline for LeWorldModel training: loads trajectory data from WebDataset shards, decodes and preprocesses frames and actions, samples subsequences, and creates PyTorch DataLoaders.

## Requirements

### Requirement: WebDataset shard loading
The system SHALL load trajectory data from WebDataset tar shards using the `webdataset` library. Each shard SHALL contain serialized RGB frames and corresponding actions.

#### Scenario: Load shards from directory
- **WHEN** a directory path containing `.tar` shard files is provided
- **THEN** the system SHALL create a `webdataset.WebDataset` pipeline that iterates over all samples in the shards

### Requirement: Sample decoding
The system SHALL decode each sample from the WebDataset pipeline into a dictionary with keys `frames` (numpy array or tensor of shape `(T, H, W, 3)`) and `actions` (numpy array or tensor of shape `(T, A_dim)`).

#### Scenario: Decode a single sample
- **WHEN** a raw WebDataset sample is read containing encoded frames and actions
- **THEN** the system SHALL decode frames to `uint8` tensors of shape `(T, H, W, 3)` and actions to `float32` tensors of shape `(T, A_dim)`

### Requirement: Frame preprocessing
The system SHALL preprocess frames by: converting to float32, normalizing to `[0, 1]`, resizing to `(224, 224)` if needed, and transposing to `(T, 3, H, W)` channel-first format.

#### Scenario: Preprocess frames for model input
- **WHEN** raw frames of shape `(T, H, W, 3)` with `uint8` values are provided
- **THEN** the output SHALL be a `float32` tensor of shape `(T, 3, 224, 224)` with values in `[0, 1]`

### Requirement: Sequence sampling
The system SHALL sample fixed-length subsequences of length `seq_length` (configurable, default 16) from each trajectory. If a trajectory is shorter than `seq_length`, it SHALL be padded or skipped (configurable).

#### Scenario: Sample a subsequence
- **WHEN** a trajectory of length 100 is loaded with `seq_length=16`
- **THEN** the system SHALL return a random contiguous subsequence of 16 frames and corresponding actions

#### Scenario: Short trajectory handling
- **WHEN** a trajectory of length 10 is loaded with `seq_length=16` and `skip_short=True`
- **THEN** the system SHALL skip this sample

### Requirement: Batch collation
The system SHALL provide a collate function that assembles a list of samples into a batch with shapes `frames: (B, T, 3, 224, 224)` and `actions: (B, T, A_dim)`.

#### Scenario: Collate a batch
- **WHEN** a list of 4 samples each with `seq_length=16` is collated
- **THEN** the output batch SHALL have `frames` of shape `(4, 16, 3, 224, 224)` and `actions` of shape `(4, 16, A_dim)`

### Requirement: DataLoader creation
The system SHALL provide a factory function that creates a PyTorch `DataLoader` from a data directory path, with configurable batch size, number of workers, and sequence length.

#### Scenario: Create a training DataLoader
- **WHEN** `create_dataloader(data_dir, batch_size=8, num_workers=4, seq_length=16)` is called
- **THEN** the system SHALL return a `DataLoader` that yields batches of preprocessed frame-action sequences
