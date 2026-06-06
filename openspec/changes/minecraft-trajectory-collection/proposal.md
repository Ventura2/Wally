## Why

Training world models for Minecraft requires large-scale, high-quality trajectory datasets of (observation, action, reward, done) transitions. No existing pipeline in this project captures gameplay data from MineStudio in a format suitable for offline training. Building this collection system now enables future world model research (e.g., LeWorldModel-style approaches) with a reliable, reproducible data source.

## What Changes

- Introduce a **trajectory collector** that connects to a running Minecraft instance via MineStudio, captures RGB frames from the player perspective, records agent actions, and persists transitions to disk.
- Add configurable **frame skip** and **observation resizing** (224x224) to control data volume and match downstream model input requirements.
- Add a **WebDataset shard exporter** that converts raw trajectory buffers into `.tar` shards loadable by PyTorch `DataLoader` via `webdataset`.
- Add a **trajectory validation tool** to inspect shards, verify schema compliance, and report statistics (frame count, episode count, action distribution).
- Trajectories stored as sequences of transitions, each containing: observation (resized RGB), action vector, timestamp, and episode metadata (episode ID, seed, world state summary).

## Capabilities

### New Capabilities
- `trajectory-collector`: Core MineStudio integration that captures RGB observations and agent actions into an in-memory buffer with configurable frame skip and resize.
- `dataset-exporter`: Converts collected trajectory buffers into WebDataset `.tar` shards with proper key naming and metadata for efficient PyTorch loading.
- `trajectory-validator`: CLI tool to inspect, validate, and report statistics on exported trajectory shards.

### Modified Capabilities

(none)

## Impact

- **Dependencies**: Requires `minestudio`, `webdataset`, `torch` (for dataloader compatibility), `numpy`, `Pillow`/`opencv` for image processing.
- **Infrastructure**: Requires a running Minecraft server instance accessible by MineStudio. No cloud infra changes.
- **Storage**: ~100K transitions at 224x224 RGB will require significant disk space; WebDataset shards handle this efficiently.
- **APIs**: No external APIs affected. Introduces internal Python module structure under a new `src/` or `collector/` package.
