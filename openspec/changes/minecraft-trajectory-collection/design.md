## Context

This project focuses on Minecraft AI research, with papers on world models (LeWorldModel) and experience transfer (Echo). The goal is to build a data collection pipeline that captures gameplay trajectories from MineStudio for future offline world model training. No training or RL infrastructure exists yet; this change establishes the data foundation.

MineStudio provides a Python API to interact with Minecraft, exposing RGB observations and an action space. The collector must bridge MineStudio's step loop to a persistent storage format optimized for large-scale PyTorch data loading.

## Goals / Non-Goals

**Goals:**
- Capture at least 100,000 valid transitions from MineStudio gameplay sessions
- Each transition contains: resized RGB observation (224x224), action vector, timestamp, episode metadata
- Export trajectories as WebDataset `.tar` shards for efficient PyTorch `DataLoader` consumption
- Provide a validation CLI to inspect and verify exported shards
- Support configurable frame skip to control data density

**Non-Goals:**
- No model training, fine-tuning, or evaluation
- No planning or policy inference during collection
- No reinforcement learning loop
- No real-time streaming or online serving
- No multi-agent or multiplayer support

## Decisions

### 1. Observation format: JPEG-compressed RGB in WebDataset shards

**Choice**: Store each observation as a JPEG-encoded byte blob inside WebDataset `.tar` shards.

**Alternatives considered**:
- Raw NumPy arrays per frame: High fidelity but ~150KB/frame uncompressed. 100K frames = ~15GB raw.
- PNG lossless: Higher quality but 3-5x larger than JPEG and slower to encode/decode.
- HDF5/LMDB: Good for random access but poor streaming performance and harder to shard for distributed training.

**Rationale**: JPEG at quality=85 gives ~5-10KB per 224x224 frame, reducing 100K transitions to ~1GB. WebDataset shards stream sequentially which is ideal for training. JPEG decode is fast with `Pillow` or `torchvision`.

### 2. Action space: MineStudio default discrete + continuous hybrid

**Choice**: Use MineStudio's native action space as-is. Serialize each action as a JSON dict within the shard sample.

**Rationale**: MineStudio exposes both discrete (keyboard/mouse buttons) and continuous (mouse delta, camera) actions. Storing the raw action dict preserves full information. Downstream consumers can discretize or filter as needed. This avoids premature commitment to a specific action encoding.

### 3. Sharding strategy: ~1000 transitions per shard

**Choice**: Each `.tar` shard contains approximately 1000 transitions. Episode boundaries are preserved within shards (episodes are not split across shards).

**Alternatives considered**:
- One shard per episode: Simple but creates highly uneven shard sizes.
- Fixed byte-size shards (~100MB): Good for distributed training but may split episodes.

**Rationale**: ~1000 transitions per shard balances shard count (~100 shards for 100K transitions) with file size (~5-10MB each). Preserving episode boundaries simplifies downstream sequence modeling.

### 4. Frame skip: Configurable at collector level, stored as metadata

**Choice**: Frame skip is a runtime parameter of the collector (default=4). The actual skip value is recorded in episode metadata so downstream consumers know the temporal resolution.

**Rationale**: Different collection runs may use different skip values. Storing it as metadata ensures reproducibility without hardcoding.

### 5. Module structure

**Choice**:
```
src/
  collector/
    __init__.py
    env.py          # MineStudio wrapper
    recorder.py     # Frame capture + action logging
    buffer.py       # In-memory trajectory buffer
    config.py       # Dataclass-based configuration
  exporter/
    __init__.py
    shard_writer.py # WebDataset shard creation
    metadata.py     # Shard index and dataset manifest
  validator/
    __init__.py
    cli.py          # CLI entry point
    inspector.py    # Shard validation logic
```

**Rationale**: Clear separation of concerns. Collector handles MineStudio interaction, exporter handles storage format, validator handles quality assurance. Each module is independently testable.

### 6. Configuration: Python dataclasses with YAML loading

**Choice**: Use `dataclasses` for config objects, loaded from a YAML file via a simple `load_config()` function.

**Alternatives considered**:
- Hydra: Powerful but heavy dependency for a data collection tool.
- argparse only: Too limited for nested configuration.

**Rationale**: Dataclasses provide type safety and defaults. YAML is human-readable and easy to version control. Minimal dependencies.

## Risks / Trade-offs

- **[MineStudio API stability]** → Pin MineStudio version in requirements. Wrap all MineStudio calls in `env.py` adapter to isolate API changes.
- **[Storage volume]** → JPEG compression keeps size manageable. Monitor disk usage during long collection runs. Add configurable max shard count as a safety limit.
- **[Episode boundary handling]** → If MineStudio resets mid-collection, the recorder must detect episode boundaries and flush the buffer. Handle `done=True` signals explicitly.
- **[JPEG quality loss]** → Acceptable for world model pretraining where exact pixel values are less critical than spatial structure. Quality=85 is a good balance. Can be made configurable.
- **[Single-threaded collection]** → Initial implementation is single-threaded (one Minecraft instance). Multi-instance parallelism is a future enhancement, not blocked by this design.
