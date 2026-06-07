# AGENTS.md

## Project

Wally is a Minecraft AI research project. Goal: train world models (LeWorldModel-style) on collected gameplay trajectories, then use them for planning (CEM-based MPC) and goal-conditioned agents. Reference papers are in the repo root as PDFs.

## Hardware

- **GPU**: AMD Radeon RX 6700 XT (RDNA 2, GFX 10.3.0)
- **ROCm**: Use `rocm/pytorch` Docker image for GPU training
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
- `src/exporter/` — WebDataset shard export (`ShardWriter`, `generate_manifest`) — legacy, used by tests
- `src/validator/` — shard inspection and validation CLI (`validator.cli` with `inspect`, `validate`, `samples`)
- `src/wally/` — LeWorldModel training pipeline
  - `models/` — ViT encoder, action embedder, causal Transformer predictor
  - `data/` — WebDataset shard loading, preprocessing, dataloader, converter
  - `training/` — losses, SIGReg, optimizer, scheduler, checkpoint, trainer, evaluation
  - `config/` — TrainConfig, ModelConfig, YAML loader
  - `cli/` — `wally-train`, `wally-convert`, `wally-collect` entry points

Tests live in `tests/` covering all packages plus an end-to-end integration test.

## CLI entry points

- `wally-collect` — collect trajectories from Minecraft, saves raw `.tar` shards to `data/raw/`
- `wally-convert` — convert raw shards to training format (`.npz` per episode) in `data/shards/`
- `wally-train` — train LeWorldModel from converted shards
- `wally-validate` — inspect/validate/sample shards

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

Note: Use TDD (Test-driven development) for `/opsx-apply` implementation tasks that contains complex logic, algorithms, or well-defined interfaces, consider writing tests first to clarify requirements before implementation. 