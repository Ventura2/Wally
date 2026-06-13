# Wally

Minecraft AI research: world models, planning, and goal-conditioned agents.

Wally collects gameplay trajectories from Minecraft (via [MineStudio](https://github.com/CraftJarvis/MineStudio)), exports them to [WebDataset](https://github.com/webdataset/webdataset) shards, validates the exported data, and trains a LeWorldModel world model — building the data pipeline for goal-conditioned agents and CEM-based MPC planning.

## Pipeline

```
Collect → Convert → Train → Play → Deploy
```

| Step | Package | What it does |
|---|---|---|
| **Collect** | `src/collector/` | Runs episodes in Minecraft via MineStudio, records observation-action-reward transitions with `frame_skip`, saves to `.tar` shards (JPEG observations + JSON sidecars). |
| **Convert** | `src/wally/data/converter.py` | Reassembles per-step shards into episode sequences (`.npz` files with frames + actions arrays) for training. |
| **Validate** | `src/validator/` | CLI + API for inspecting shard stats, validating schema/JPEG integrity, and extracting sample frames. |
| **Train** | `src/wally/` | Trains a LeWorldModel (ViT encoder + causal Transformer predictor + SIGReg) on converted shards. |
| **Play** | `src/agent/` | Runs a goal-conditioned agent loop locally via MineStudio — plan, execute, observe, repeat — with warm-start CEM replanning and trajectory recording. |
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
wally-validate samples data/shards/ --count 5 --output-dir samples/
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
# Create your own config file based on collector/config.py defaults
wally-collect --config configs/collector.yaml
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
  depth: 4
  num_heads: 4
  mlp_ratio: 4.0
  dropout: 0.1
  action_dim: 25
  pretrained: false
  encoder_type: cnn        # "cnn" (default, stable on RDNA2) or "vit" (timm ViT-Tiny)

training:
  lr: 0.0001
  weight_decay: 0.00001
  warmup_steps: 500
  max_steps: 100000
  batch_size: 16
  seq_length: 16
  alpha: 0.1              # SIGReg loss weight (LeWM paper Section 3.1)
  sigreg_num_proj: 1024   # random projections for closed-form SIGReg
  sigreg_knots: 17        # knots for Epps-Pulley integration grid
  use_amp: true            # mixed precision (bfloat16 by default, fp16 optional)
  amp_dtype: bfloat16      # autocast dtype; use "float16" for GradScaler path
  checkpoint_interval: 1000
  log_interval: 100
  data_dir: data/shards/chunks
  output_dir: checkpoints
  num_workers: 8
  persistent_workers: true
  prefetch_factor: 4
  skip_short: true         # skip trajectories shorter than seq_length
  wandb_project: wally
```

#### Training output

| Output | Location | Description |
|---|---|---|
| **Checkpoints** | `checkpoints/checkpoint_<step>.pt` | Model + optimizer + scheduler state dicts, step count, config. Saved every `checkpoint_interval` steps and at end of training. |
| **Wandb logs** | W&B dashboard | Prediction loss, SIGReg loss, total loss, learning rate — logged every `log_interval` steps. Set `wandb_project` in config. |

#### Training stability

The trainer applies a NaN/Inf guard: if `total_loss` is non-finite on a step, the optimizer update is skipped, a warning is logged, and `global_step` advances. Input batches are also sanitized with `torch.nan_to_num` before the forward pass. SIGReg uses the closed-form Epps-Pulley statistic (Epps & Pulley, 1983) on `num_proj` random unit-norm projections of the encoder embeddings — stateless, non-negative, finite for any finite input.

#### Checkpoint contents

Each checkpoint is a `.pt` file containing:
- `model_state_dict` — LeWorldModel weights
- `optimizer_state_dict` — AdamW optimizer state
- `scheduler_state_dict` — LR scheduler state (cosine + warmup `last_epoch`); on resume, the LR schedule continues from this state instead of restarting warmup
- `global_step` — training step at save time
- `config` — full training config dict

Checkpoints saved before the `lewm-adaln-predictor` change use a different model architecture (interleaved-input TransformerEncoder with default-affine LayerNorms) and cannot be loaded by the current code. They are archived in `checkpoints/_incompatible_pre_adaln/`. All new runs start from step 0.

### Step 4: Plan

Plan action sequences using a trained world model. Two planning modes are available:

#### Flat planning (CEM-based MPC)

Short-horizon planning using Cross-Entropy Method optimization:

```bash
wally-plan --checkpoint checkpoints/model.pt --frames frames_dir/ --output actions.pt
```

#### Hierarchical planning

Long-horizon planning with automatic subgoal decomposition:

```bash
wally-plan-hierarchical \
    --checkpoint checkpoints/model.pt \
    --high-level-checkpoint checkpoints/high_level.pt \
    --frames frames_dir/ \
    --output plan.pt
```

The hierarchical planner:
- Detects context-change points in trajectories using prediction error analysis
- Trains a high-level world model on abstract transitions between subgoals
- Plans sequences of latent subgoals toward distant goals
- Executes subgoals sequentially with low-level planner, with replanning on failure
- Supports gradient-based MPC refinement and ensemble uncertainty estimation

#### Curriculum training

Train with progressive horizon increases for better long-horizon learning:

```bash
wally-train-curriculum \
    --data-dir data/shards \
    --output-dir checkpoints \
    --stages 8,16,32,64 \
    --loss-threshold 0.01 \
    --patience 5
```

### Step 5: Play (local agent loop)

Run a goal-conditioned agent locally via MineStudio. The agent plans, executes, observes, and replans in a loop:

```bash
wally-play --checkpoint checkpoints/model.pt \
    --goal-frame goals/collect_wood.png \
    --record --output-dir data/recordings
```

The agent loop:
- Plans action sequences using a trained world model (CEM-based MPC)
- Executes actions in the environment at fixed intervals
- Replans with warm-start (shifts previous plan, reuses CEM samples)
- Records trajectories for analysis or retraining
- Supports both flat and hierarchical planners (`--planner hierarchical`)

### Step 6: Deploy

Once trained, deploy the agent to play Minecraft autonomously on a live server.

#### Basic usage

Deploy to a local offline-mode server:

```bash
wally-deploy --server localhost:25565 \
    --checkpoint checkpoints/checkpoint_10000.pt \
    --goal-frame goals/collect_wood.png
```

Deploy to an online-mode server with Microsoft authentication:

```bash
wally-deploy --server play.example.com:25565 \
    --checkpoint checkpoints/checkpoint_10000.pt \
    --goal-frame goals/collect_wood.png
```

The agent will:
- Connect to the server and authenticate
- Reconstruct first-person observations from chunk data
- Plan actions using the trained world model
- Execute actions at 20 TPS with safety filters
- Automatically reconnect on disconnect
- Save state for session persistence

#### Recording trajectories

Record the agent's gameplay for analysis or retraining:

```bash
wally-deploy --server localhost:25565 \
    --checkpoint checkpoints/checkpoint_10000.pt \
    --goal-frame goals/collect_wood.png \
    --record \
    --output-dir data/recordings
```

**Output**: `data/recordings/episode_0.npz` — trajectory files in NumPy format.

#### Configuration

Use a YAML config for advanced options:

```yaml
# deploy_config.yaml
server_host: localhost
server_port: 25565
auth_mode: offline  # or "online" for Microsoft auth
username: WallyAgent
checkpoint_path: checkpoints/checkpoint_10000.pt
goal_frame_path: goals/collect_wood.png
render_distance: 4

safety:
  prevent_bedrock_breaking: true
  prevent_lava_interaction: true
  prevent_void_fall: true1
  void_threshold: -64.0
  action_cooldown_ms: 100

reconnect:
  max_attempts: 10
  initial_backoff_s: 1.0
  max_backoff_s: 60.0

log_dir: logs/deploy
log_to_stdout: false
record_trajectory: false
output_dir: data/recordings
```

```bash
wally-deploy --config deploy_config.yaml
```

CLI arguments override config file values:

```bash
wally-deploy --config deploy_config.yaml --server prod.example.com:25565
```

#### Features

- **Network protocol**: pyCraft (Minecraft Java Edition 1.8-1.20+)
- **Authentication**: Offline mode (username-only) or online mode (Microsoft OAuth with token caching)
- **Automatic reconnection**: Exponential backoff (1s → 2s → 4s → ... → 60s max), up to 10 attempts
- **Action throttling**: 20 TPS rate limiting with adaptive timing for lagging servers
- **Safety filters**: Bedrock breaking prevention, lava interaction prevention, void fall prevention, action cooldowns
- **Session persistence**: Position, inventory, and goal progress saved to checkpoint file
- **Structured logging**: JSON logs with rotation, action tracking, position monitoring
- **Graceful shutdown**: SIGINT/SIGTERM handlers save state before disconnecting

#### Which deployment path to choose?

- **Local evaluation** (via `wally-play`): Use for development, testing, and benchmarking in MineStudio's local environment
- **Live server deployment** (via `wally-deploy`): Use for real Minecraft servers, multi-player environments, and persistent autonomous gameplay

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
  deployer/      # server connector, auth, session manager, action throttler, executor, frame renderer, safety filters, ServerEnv adapter, logging, shutdown, CLI
  exporter/      # ShardWriter, manifest generation (legacy, used by tests)
  validator/     # shard inspection, validation, sample extraction
  wally/         # LeWorldModel training pipeline
    models/      # ViT encoder, action embedder, causal Transformer predictor, recurrent encoder
    data/        # WebDataset shard loading, preprocessing, dataloader, converter
    training/    # losses, SIGReg, optimizer, scheduler, checkpoint, trainer, evaluation, curriculum, curiosity, ensemble
    config/      # TrainConfig, ModelConfig, YAML loader
    planner/     # CEM, latent rollout, goal-conditioned planner, gradient MPC, subgoal detector, high-level planner, hierarchical planner
    cli/         # wally-train, wally-convert, wally-collect, wally-train-curriculum entry points
  agent/         # goal-conditioned agent loop (env adapter, planner protocol, trajectory buffer, agent loop, play CLI)
configs/         # example YAML configs
tests/           # unit tests + end-to-end integration test
```

## References

- [LeWorldModel: Echo — Experience Transfer for Multimodal LLM Agents in Minecraft Game](./LeWorldModel.pdf)
- [Optimus-3: Foundation Model for Minecraft](./optimus-3.pdf)
- [Echo: Experience Transfer for Multimodal LLM Agents in Minecraft Game](./Echo%20-%20Experience%20Transfer%20for%20Multimodal%20LLM%20Agents%20in%20Minecraft%20Game.pdf)
