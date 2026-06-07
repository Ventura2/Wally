# Wally

Minecraft AI research: world models, planning, and goal-conditioned agents.

Wally collects gameplay trajectories from Minecraft (via [MineStudio](https://github.com/CraftJarvis/MineStudio)), exports them to [WebDataset](https://github.com/webdataset/webdataset) shards, validates the exported data, and trains a LeWorldModel world model — building the data pipeline for goal-conditioned agents and CEM-based MPC planning.

## Pipeline

```
Collect → Export → Validate → Train
```

| Step | Package | What it does |
|---|---|---|
| **Collect** | `src/collector/` | Runs episodes in Minecraft via MineStudio, records observation-action-reward transitions with `frame_skip`, accumulates them in a buffer. |
| **Export** | `src/exporter/` | Writes transitions to `.tar` shards (JPEG observations + JSON sidecars) and generates a `manifest.json`. |
| **Validate** | `src/validator/` | CLI + API for inspecting shard stats, validating schema/JPEG integrity, and extracting sample frames. |
| **Train** | `src/wally/` | Trains a LeWorldModel (ViT encoder + causal Transformer predictor + SIGReg) on exported shards. |

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

### Train LeWorldModel

Training reads WebDataset shards from `data/shards/` by default and writes checkpoints to `checkpoints/`.

```bash
# Train with default config
wally-train --config configs/lewm_default.yaml

# Train on CPU
wally-train --config configs/lewm_default.yaml --device cpu

# Resume from checkpoint
wally-train --config configs/lewm_default.yaml --resume checkpoints/checkpoint_5000.pt
```

#### Custom config

Copy and edit `configs/lewm_default.yaml`:

```yaml
model:
  vit_variant: vit_tiny_patch16_224
  embed_dim: 192
  depth: 6
  num_heads: 4
  mlp_ratio: 4.0
  dropout: 0.1
  action_dim: 25
  pretrained: true

training:
  lr: 0.0001
  weight_decay: 0.00001
  warmup_steps: 1000
  max_steps: 100000
  batch_size: 8
  seq_length: 16
  alpha: 0.1              # SIGReg loss weight
  use_amp: false           # mixed precision (fp16)
  checkpoint_interval: 1000
  log_interval: 10
  data_dir: data/shards
  output_dir: checkpoints
  num_workers: 4
  skip_short: true         # skip trajectories shorter than seq_length
  wandb_project: wally
```

#### Training output

| Output | Location | Description |
|---|---|---|
| **Checkpoints** | `checkpoints/checkpoint_<step>.pt` | Model + optimizer + SIGReg critic state dicts, step count, config. Saved every `checkpoint_interval` steps and at end of training. |
| **Wandb logs** | W&B dashboard | Prediction loss, SIGReg loss, total loss, learning rate — logged every `log_interval` steps. Set `wandb_project` in config. |

#### Checkpoint contents

Each checkpoint is a `.pt` file containing:
- `model_state_dict` — LeWorldModel weights
- `optimizer_state_dict` — AdamW optimizer state
- `critic_optimizer_state_dict` — SIGReg critic optimizer state
- `global_step` — training step at save time
- `config` — full training config dict

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
  wally/         # LeWorldModel training pipeline
    models/      # ViT encoder, action embedder, causal Transformer predictor
    data/        # WebDataset shard loading, preprocessing, dataloader
    training/    # losses, SIGReg, optimizer, scheduler, checkpoint, trainer, evaluation
    config/      # TrainConfig, ModelConfig, YAML loader
    cli/         # wally-train entry point
configs/         # example YAML configs
tests/           # unit tests + end-to-end integration test
```

## References

- [LeWorldModel: Echo — Experience Transfer for Multimodal LLM Agents in Minecraft Game](./LeWorldModel.pdf)
- [Optimus-3: Foundation Model for Minecraft](./optimus-3.pdf)
- [Echo: Experience Transfer for Multimodal LLM Agents in Minecraft Game](./Echo%20-%20Experience%20Transfer%20for%20Multimodal%20LLM%20Agents%20in%20Minecraft%20Game.pdf)
