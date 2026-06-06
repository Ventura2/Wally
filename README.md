# Wally

Minecraft AI research: world models, planning, and goal-conditioned agents.

Wally collects gameplay trajectories from Minecraft (via [MineStudio](https://github.com/CraftJarvis/MineStudio)), exports them to [WebDataset](https://github.com/webdataset/webdataset) shards, and validates the exported data — building the data pipeline for training world models (inspired by [LeWorldModel](https://arxiv.org/abs/2311.15234)) and CEM-based MPC planning.

## Pipeline

```
Collect → Export → Validate
```

| Step | Package | What it does |
|---|---|---|
| **Collect** | `src/collector/` | Runs episodes in Minecraft via MineStudio, records observation-action-reward transitions with `frame_skip`, accumulates them in a buffer. |
| **Export** | `src/exporter/` | Writes transitions to `.tar` shards (JPEG observations + JSON sidecars) and generates a `manifest.json`. |
| **Validate** | `src/validator/` | CLI + API for inspecting shard stats, validating schema/JPEG integrity, and extracting sample frames. |

## Setup

```bash
uv sync                  # install dependencies
```

## Usage

### Collect trajectories

```python
from collector.collector import TrajectoryCollector
from collector.config import CollectorConfig

config = CollectorConfig(num_episodes=10)
collector = TrajectoryCollector(config)
transitions = collector.run()
```

### Export to WebDataset

```python
from exporter.shard_writer import ShardWriter
from exporter.metadata import generate_manifest

writer = ShardWriter(output_dir="shards", max_transitions_per_shard=1000)
writer.write_all(transitions)
generate_manifest("shards")
```

### Validate shards

```bash
python -m validator.cli inspect shards/
python -m validator.cli validate shards/
python -m validator.cli samples shards/ --num 5 --output samples/
```

## Tests

```bash
uv run pytest                 # full suite
uv run pytest tests/<file> -k "<name>"   # specific test
```

## Lint & typecheck

```bash
uv run ruff check .
uv run mypy
```

## Project structure

```
src/
  collector/     # env wrapper, recorder, buffer, config, orchestrator
  exporter/      # ShardWriter, manifest generation
  validator/     # shard inspection, validation, sample extraction
tests/           # unit tests + end-to-end integration test
```

## References

- [LeWorldModel: Echo — Experience Transfer for Multimodal LLM Agents in Minecraft Game](./LeWorldModel.pdf)
- [Optimus-3: Foundation Model for Minecraft](./optimus-3.pdf)
- [Echo: Experience Transfer for Multimodal LLM Agents in Minecraft Game](./Echo%20-%20Experience%20Transfer%20for%20Multimodal%20LLM%20Agents%20in%20Minecraft%20Game.pdf)
