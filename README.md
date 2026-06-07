# Wally

Minecraft AI research: world models, planning, and goal-conditioned agents.

Wally collects gameplay trajectories from Minecraft (via [MineStudio](https://github.com/CraftJarvis/MineStudio)), exports them to [WebDataset](https://github.com/webdataset/webdataset) shards, validates the exported data, and trains a LeWorldModel world model — building the data pipeline for goal-conditioned agents and CEM-based MPC planning.

## Pipeline

```
Collect → Convert → Train → Deploy
```

| Step | Package | What it does |
|---|---|---|
| **Collect** | `src/collector/` | Runs episodes in Minecraft via MineStudio, records observation-action-reward transitions with `frame_skip`, saves to `.tar` shards (JPEG observations + JSON sidecars). |
| **Convert** | `src/wally/data/converter.py` | Reassembles per-step shards into episode sequences (`.npz` files with frames + actions arrays) for training. |
| **Validate** | `src/validator/` | CLI + API for inspecting shard stats, validating schema/JPEG integrity, and extracting sample frames. |
| **Train** | `src/wally/` | Trains a LeWorldModel (ViT encoder + causal Transformer predictor + SIGReg) on converted shards. |
| **Deploy** | `src/deployer/` | Runs trained agent on Minecraft — locally via MineStudio or on a live server via network protocol. |

## Concepts

### Episodes

An **episode** is one complete gameplay session — from when the agent spawns in Minecraft until it dies or the session ends. It contains a sequence of transitions (frame + action pairs).

When you run `wally-collect --episodes 10`, it plays 10 full games and records all the frames/actions from each.

### Shards

A **shard** is a `.tar` archive that bundles multiple episodes together for efficient I/O. Instead of storing millions of individual files, you pack them into tarballs:

```
data/raw/shard_000000.tar          # Raw format (per-step)
  ep1_000000.jpg + ep1_000000.json
  ep1_000001.jpg + ep1_000001.json
  ep2_000000.jpg + ep2_000000.json
  ...

data/shards/shard_000000.tar       # Training format (per-episode)
  ep1.npz (contains all frames + actions for episode 1)
  ep2.npz (contains all frames + actions for episode 2)
  ...
```

**Why shards?**
- **Fast I/O**: Reading sequentially from a tar is 3-10x faster than random file access
- **Streaming**: Training can start before all data is loaded
- **Scalable**: Works for datasets from GBs to TBs

## Setup

```bash
uv sync                  # install dependencies
```

## Usage

### Collect trajectories

```bash
wally-collect --episodes 10 --output-dir data/raw
```

### Convert to training format

```bash
wally-convert --input data/raw --output data/shards --config configs/converter_default.yaml
```

### Validate shards

```bash
wally-validate inspect data/shards/
wally-validate validate data/shards/
wally-validate samples data/shards/ --num 5 --output samples/
```

## Running the full pipeline

### Prerequisites

Choose one of these options:

**Option A: Podman/Docker (recommended)**
- Podman with podman-compose (or Docker with docker-compose)
- AMD GPU: ROCm runtime (`/dev/kfd`, `/dev/dri`)
- NVIDIA GPU: nvidia-container-toolkit

**Option B: WSL2 (Linux)**
- WSL2 with Python 3.12+
- AMD GPU (ROCm) or NVIDIA GPU (CUDA) for training


Both options require a Minecraft Java server running and accessible.

### Step 0: Environment setup

#### Using Podman/Docker (recommended)

```bash
# Clone and enter directory
git clone <repo-url> wally
cd wally

# Build and start container
podman-compose up --build -d

# Enter the container
podman exec -it wally-dev bash

# Inside container: install dependencies
uv sync

# Verify installation
uv run pytest
```

Or use the helper script:
```bash
# Windows
docker-run.bat

# Linux/Mac
./docker-run.sh
```

The container mounts your local repo at `/workspace`, so code changes are reflected immediately without rebuilding.

#### Using WSL2

```bash
# In WSL2
git clone <repo-url> wally
cd wally
uv sync
uv pip install minestudio

# Verify installation
uv run pytest
```

### Step 1: Collect trajectories

Use the `wally-collect` CLI to run episodes in Minecraft and record transitions:

```bash
# Quick test: 1 episode
wally-collect --episodes 1 --output-dir data/raw

# Full run: 100 episodes with custom settings
wally-collect --episodes 100 --output-dir data/raw \
    --frame-skip 4 --resize 224 224 --buffer-size 10000
```

Or via a YAML config:

```bash
wally-collect --config configs/collector_default.yaml
```

**Requirements**: Minecraft Java server must be running and accessible (default `localhost:25565`).

**Output**: `data/raw/*.tar` — each contains per-step JPEG frames + JSON action sidecars.

### Step 2: Convert to training format

The training pipeline expects episode sequences (`.npz` files), not per-step data. Convert with:

```bash
wally-convert --input data/raw --output data/shards --config configs/converter_default.yaml
```

Or programmatically:

```python
from src.wally.data.converter import convert_shards

convert_shards(
    input_dir="data/raw",
    output_dir="data/shards",
    action_schema=["forward", "backward", "left", "right", "jump", ...],  # 25 keys
    episodes_per_shard=50,
)
```

**Output**: `data/shards/*.tar` — each contains `.npz` files with `frames` (T, H, W, 3) and `actions` (T, 25) arrays.

### Step 3: Train

```bash
wally-train --config configs/lewm_default.yaml
```

Training reads from `data/shards/` (configured in `lewm_default.yaml`).

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

### Step 4: Deploy

Once trained, deploy the agent to play Minecraft autonomously. Two deployment paths are available:

#### Path A: Local Evaluation (MineStudio)

Run the agent in a local MineStudio environment for development, testing, and benchmarking.

```bash
wally-deploy --checkpoint checkpoints/checkpoint_10000.pt --mode local --goal "collect_wood"
```

**Use when:**
- Developing and tuning the planner
- Running benchmarks and evaluations
- Debugging agent behavior
- Fast iteration without server setup

**Features:**
- Direct environment access (no network latency)
- Synchronous execution
- Full access to environment state
- Reproducible results

#### Path B: Live Server Deployment

Deploy the agent to a live Minecraft server (vanilla, Paper, Spigot, Fabric) for persistent, multi-player gameplay.

```bash
wally-deploy --checkpoint checkpoints/checkpoint_10000.pt --mode server \
    --server localhost:25565 --username WallyAgent --offline
```

Or with Microsoft account authentication:

```bash
wally-deploy --checkpoint checkpoints/checkpoint_10000.pt --mode server \
    --server play.example.com --auth microsoft
```

**Use when:**
- Watching the agent play in real-time (spectate from the server)
- Multi-player environments
- Persistent autonomous gameplay
- Demonstrating agent capabilities

**Features:**
- Network protocol via pyCraft (Minecraft 1.8-1.20+)
- Automatic reconnection on disconnect
- Action throttling (20 TPS)
- Safety bounds (configurable)
- Session persistence and state recovery

**Configuration:**

Use a YAML config for advanced options:

```yaml
# deploy_config.yaml
mode: server
checkpoint: checkpoints/checkpoint_10000.pt
server:
  address: localhost:25565
  auth: offline  # or "microsoft"
  username: WallyAgent
agent:
  goal: "collect_wood"
  replan_interval: 20  # steps
safety:
  prevent_bedrock_break: true
  prevent_lava_interaction: true
  action_cooldown_ms: 50
```

```bash
wally-deploy --config deploy_config.yaml
```

**Which path to choose?**

Start with **Path A (Local)** for development and testing. When ready to see the agent play in a real Minecraft world or interact with other players, deploy to a **Path B (Live Server)**.

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
  collector/     # env wrapper, recorder, buffer, config, raw_shard_writer
  deployer/      # server connector, session manager, action throttler, safety filter
  exporter/      # ShardWriter, manifest generation (legacy, used by tests)
  validator/     # shard inspection, validation, sample extraction
  wally/         # LeWorldModel training pipeline
    models/      # ViT encoder, action embedder, causal Transformer predictor
    data/        # WebDataset shard loading, preprocessing, dataloader, converter
    training/    # losses, SIGReg, optimizer, scheduler, checkpoint, trainer, evaluation
    config/      # TrainConfig, ModelConfig, YAML loader
    cli/         # wally-train, wally-convert, wally-collect, wally-deploy entry points
configs/         # example YAML configs
tests/           # unit tests + end-to-end integration test
```

## References

- [LeWorldModel: Echo — Experience Transfer for Multimodal LLM Agents in Minecraft Game](./LeWorldModel.pdf)
- [Optimus-3: Foundation Model for Minecraft](./optimus-3.pdf)
- [Echo: Experience Transfer for Multimodal LLM Agents in Minecraft Game](./Echo%20-%20Experience%20Transfer%20for%20Multimodal%20LLM%20Agents%20in%20Minecraft%20Game.pdf)
