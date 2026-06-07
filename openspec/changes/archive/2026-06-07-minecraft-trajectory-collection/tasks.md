## 1. Project Setup

- [x] 1.1 Create module directory structure (`src/collector/`, `src/exporter/`, `src/validator/`) with `__init__.py` files
- [x] 1.2 Create `pyproject.toml` or `requirements.txt` with dependencies: `minestudio`, `webdataset`, `torch`, `numpy`, `Pillow`, `pyyaml`
- [x] 1.3 Create `src/collector/config.py` with dataclass-based configuration (frame_skip, resize dimensions, jpeg_quality, buffer_size, output_dir) and YAML loading

## 2. Trajectory Collector

- [x] 2.1 Implement `src/collector/env.py` — MineStudio environment wrapper that initializes the connection, exposes `step()`, `reset()`, and returns raw RGB observations
- [x] 2.2 Implement observation resize pipeline in `env.py` — resize raw frames to 224x224 using bilinear interpolation via Pillow
- [x] 2.3 Implement `src/collector/recorder.py` — capture RGB observation and action at each step, attach millisecond Unix timestamp, pair them into a transition dict
- [x] 2.4 Implement frame skip logic in `recorder.py` — execute N environment steps per recorded transition, accumulate reward, store only the final observation
- [x] 2.5 Implement episode metadata tracking in `recorder.py` — assign unique `episode_id` per episode, record `seed`, detect `done=True` boundaries
- [x] 2.6 Implement `src/collector/buffer.py` — in-memory trajectory buffer with configurable max size, flush callback interface, and graceful shutdown flush
- [x] 2.7 Implement collector main loop that wires env, recorder, and buffer together into a runnable collection session

## 3. Dataset Exporter

- [x] 3.1 Implement `src/exporter/shard_writer.py` — accept a list of transitions and write them to a `.tar` file in WebDataset format
- [x] 3.2 Implement JPEG observation encoding in `shard_writer.py` — encode each observation as JPEG at configurable quality (default 85), stored as `{key}.jpg`
- [x] 3.3 Implement JSON sidecar encoding in `shard_writer.py` — serialize action dict, timestamp, episode_id, step_index, frame_skip, seed into `{key}.json`
- [x] 3.4 Implement shard key naming convention: `{episode_id}_{step_index:06d}`
- [x] 3.5 Implement episode-boundary-aware sharding — keep episodes within a single shard, target ~1000 transitions per shard
- [x] 3.6 Implement `src/exporter/metadata.py` — generate `manifest.json` with total transitions, total episodes, shard count, shard file list, and per-shard transition counts
- [x] 3.7 Implement output directory auto-creation in the exporter

## 4. Trajectory Validator

- [x] 4.1 Implement `src/validator/inspector.py` — read a `.tar` shard and extract transition count, episode count, observation shape, action keys, timestamp range
- [x] 4.2 Implement schema validation in `inspector.py` — verify every `.jpg` has a matching `.json`, detect corrupt JPEG files, report errors with sample keys
- [x] 4.3 Implement action distribution computation — per-action-key mean/std/min/max for continuous values, value counts for discrete values
- [x] 4.4 Implement minimum transition count check against 100,000 threshold
- [x] 4.5 Implement sample visualization — extract N random observations, decode JPEGs, save as PNG files to a specified output directory
- [x] 4.6 Implement `src/validator/cli.py` — CLI entry point with subcommands: `inspect`, `validate`, `samples`; wire to inspector functions; set exit codes (0=pass, 1=fail)

## 5. Integration and Testing

- [x] 5.1 Write unit tests for observation resize (verify output shape is 224x224)
- [x] 5.2 Write unit tests for frame skip logic (verify correct number of steps executed per transition)
- [x] 5.3 Write unit tests for buffer flush behavior (threshold flush and shutdown flush)
- [x] 5.4 Write unit tests for shard writer (verify tar contents, key naming, JPEG decodability, JSON parseability)
- [x] 5.5 Write unit tests for manifest generation (verify all fields present and counts correct)
- [x] 5.6 Write unit tests for validator (valid shard passes, missing sidecar fails, corrupt JPEG fails)
- [x] 5.7 Write integration test: run collector with a mock MineStudio environment, export shards, validate output with the validator CLI
