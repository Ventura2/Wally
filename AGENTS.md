# AGENTS.md

## Project

Wally is a Minecraft AI research project. Goal: train world models (LeWorldModel-style) on collected gameplay trajectories, then use them for planning (CEM-based MPC) and goal-conditioned agents. Reference papers are in the repo root as PDFs.

## Hardware

- **GPU**: AMD Radeon RX 6700 XT (RDNA 2, GFX 10.3.0)
- **ROCm**: Use `rocm/pytorch` Podman image for GPU training
- **Dev container**: Podman with ROCm device passthrough (`/dev/kfd`, `/dev/dri`)

## Setup commands

- `uv sync` — install all dependencies
- `uv run pytest` — run the full test suite
- `uv run pytest tests/<file> -k "<name>"` — run a specific test
- `uv run ruff check .` — lint
- `uv run mypy` — typecheck

## Project structure

Application code lives in `src/` with these packages:
- `src/collector/` — trajectory collection (env wrapper, buffer, recorder, config, `raw_shard_writer`)
- `src/deployer/` — Minecraft server deployment (connector, auth, session manager, action throttler, action executor, frame renderer, safety filters, ServerEnv adapter, logging, graceful shutdown, CLI)
- `src/exporter/` — WebDataset shard export (`ShardWriter`, `generate_manifest`) — legacy, used by tests
- `src/validator/` — shard inspection and validation CLI (`validator.cli` with `inspect`, `validate`, `samples`)
- `src/wally/` — LeWorldModel training pipeline
  - `models/` — ViT encoder, action embedder, causal Transformer predictor, recurrent encoder
  - `data/` — WebDataset shard loading, preprocessing, dataloader, converter
  - `training/` — losses, SIGReg, optimizer, scheduler, checkpoint, trainer, evaluation, curriculum, curiosity, ensemble
  - `config/` — TrainConfig, ModelConfig, YAML loader
  - `planner/` — CEM optimizer, latent rollout, goal-conditioned planner, gradient MPC, subgoal detector, high-level planner, hierarchical planner
  - `cli/` — `wally-train`, `wally-convert`, `wally-collect`, `wally-train-curriculum` entry points
- `src/agent/` — goal-conditioned agent loop (env adapter, planner protocol, trajectory buffer, agent loop, play CLI)

Tests live in `tests/` covering all packages plus an end-to-end integration test.

## CLI entry points

- `wally-collect` — collect trajectories from Minecraft, saves raw `.tar` shards to `data/raw/`
- `wally-convert` — convert raw shards to training format (`.npz` per episode) in `data/shards/`
- `wally-train` — train LeWorldModel from converted shards
- `wally-train-curriculum` — train with progressive horizon curriculum (8 → 16 → 32 → full)
- `wally-plan` — plan action sequences using CEM-based MPC
- `wally-plan-hierarchical` — hierarchical planning with subgoal decomposition
- `wally-play` — run goal-conditioned agent loop locally via MineStudio (plan, execute, observe, replan)
- `wally-validate` — inspect/validate/sample shards
- `wally-deploy` — deploy trained agent to a Minecraft server (connection, auth, action execution, safety filters)

## GPU setup (WSL2 + ROCm + librocdxg)

The RX 6700 XT (RDNA2) is **not** in AMD's official WSL2 ROCm compatibility matrix (which only lists RDNA3/RDNA4 and Ryzen AI APUs). However, it **does work** with a custom `dids.conf` entry. The setup requires building AMD's open-source `librocdxg` library from source.

### Prerequisites

- **Windows**: AMD Adrenalin driver (≥ 26.2.2 for WSL GPU-P support), Windows SDK 10.0.26100.0
- **WSL2 Ubuntu 24.04**: ROCm 7.2.x, GCC ≥ 11.4, CMake ≥ 3.15
- Windows SDK must be installed at `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\`

### Build librocdxg

```sh
# Clone the repo (to a persistent path, NOT /tmp which gets cleaned)
git clone https://github.com/ROCm/librocdxg.git ~/librocdxg
cd ~/librocdxg

# Create a cmake toolchain file to pass WIN_SDK (path has spaces, breaks -D flag)
cat > ~/rocdxg_toolchain.cmake << 'EOF'
set(WIN_SDK "/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared")
EOF

# Configure and build
mkdir -p build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/rocdxg_toolchain.cmake
make -j$(nproc)
sudo make install
```

### Configure dids.conf for RX 6700 XT

The `dids.conf` at `/opt/rocm/share/rocdxg/dids.conf` allows adding unsupported device IDs:

```
# Add this line (device_id, gfx_major, gfx_minor, gfx_stepping)
0x73DF,10,3,1    # Radeon RX 6700 XT, gfx1031
```

### Enable GPU detection

```sh
# Add to ~/.bashrc for persistence
echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verify

```sh
rocminfo
# Look for Agent with: gfx1031, AMD Radeon RX 6700 XT, Chip ID 0x73df
```

### Known issues

- `rocm-smi` is a Python script and may fail if `python3` isn't on PATH in the shell
- `dmesg` will show `dxgkio_query_adapter_info: Ioctl failed: -22` — this is from the amdgpu kernel module and is **benign** (userspace path works fine)
- `rocminfo` shows `Warning: Windows driver is old` — this is a non-fatal warning

## Running the collector (Podman container)

The `wally-collect` CLI entry point uses `uv run` — but inside the `rocm/pytorch` container, `uv sync` does not work because the container has Python 3.10 while `pyproject.toml` requires `>=3.12`. Run directly with the system Python instead:

```sh
# Inside the container (wally-dev)
podman exec wally-dev sh -c 'cd /workspace && PYTHONPATH=src python3 -m wally.cli.collect --episodes 1 --output-dir data/raw --max-steps 5000'
```

Key details:
- Uses system Python 3.10 and system-installed `minestudio` at `/usr/local/lib/python3.10/dist-packages/`
- Requires `PYTHONPATH=src` because the `src/`-layout package isn't installed via pip in the container
- `--max-steps` prevents infinite episodes (the `HumanSurvival` task only ends on player death, which may never happen with random actions)
- The Minecraft engine fat jar is at `/tmp/MineStudio/engine/build/libs/mcprec-6.13.jar` (downloaded by `python -m minestudio.simulator.entry -y`)
- A symlink from `MCP-Reborn/build/libs/mcprec-6.13.jar` → engine jar exists at `/workspace/.venv/Lib/site-packages/minestudio/simulator/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar`
- Known benign warnings: `fliteWrapper` library, `optifine/ctm/default/empty.png` texture, OpenAL sound device, Realms auth — all safe to ignore

## Data format

- **Raw shards** (`data/raw/*.tar`): per-step JPEG frames + JSON action sidecars
- **Training shards** (`data/shards/*.tar`): per-episode `.npz` files with `frames` (T,H,W,3) and `actions` (T,25) arrays

## Code style

- Python 3.12+, follows PEP 8
- Match patterns from existing code
- `pyproject.toml` is the single source of truth for dependencies and tool config

## Testing instructions

- Run the full suite with `uv run pytest` before every commit
- Add or update tests for any code you change
- The integration test (`tests/test_integration.py`) runs the full collect → convert → validate pipeline with a mock environment

## OpenSpec workflow

All feature work goes through OpenSpec. Config: `openspec/config.yaml`.

Commands:
- `/opsx-propose <name>` — create a change with proposal, design, specs, and tasks
- `/opsx-apply` — implement tasks from a change
- `/opsx-archive` — archive a completed change
- `/opsx-explore` — think through ideas before committing to a change
- `/opsx-sync` — sync delta specs from a change into main specs

Key directories:
- `openspec/specs/` — main capability specs (shared across changes, source of truth)
- `openspec/changes/` — active changes with delta specs, designs, and task lists
- `openspec/changes/archive/` — completed changes

## /opsx-apply workflow

When running `/opsx-apply`, delegate each task to a separate subagent via the Task tool instead of implementing tasks serially in the main conversation. This keeps each task focused and parallelizable.

Pattern for each task:
1. Read the task description and all relevant context files (specs, design, existing code)
2. Create a subagent via the Task tool with a detailed prompt covering: what to implement, which files to edit/create, how to verify (lint, typecheck, test commands), and relevant code conventions
3. The subagent returns when done — review the result, ensure tests pass, then mark the task complete in the tasks file
4. Move to the next task

Use the `general` subagent type for implementation tasks. Use the `explore` subagent type for research/investigation tasks.

Note: Use TDD (test-driven development) for tasks that contain complex logic, algorithms, or well-defined interfaces — write tests first to clarify requirements before implementation. 