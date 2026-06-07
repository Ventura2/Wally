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
  - `models/` — ViT encoder, action embedder, causal Transformer predictor, recurrent encoder
  - `data/` — WebDataset shard loading, preprocessing, dataloader, converter
  - `training/` — losses, SIGReg, optimizer, scheduler, checkpoint, trainer, evaluation, curriculum, curiosity, ensemble
  - `config/` — TrainConfig, ModelConfig, YAML loader
  - `planner/` — CEM optimizer, latent rollout, goal-conditioned planner, gradient MPC, subgoal detector, high-level planner, hierarchical planner
  - `cli/` — `wally-train`, `wally-convert`, `wally-collect`, `wally-train-curriculum` entry points

Tests live in `tests/` covering all packages plus an end-to-end integration test.

## CLI entry points

- `wally-collect` — collect trajectories from Minecraft, saves raw `.tar` shards to `data/raw/`
- `wally-convert` — convert raw shards to training format (`.npz` per episode) in `data/shards/`
- `wally-train` — train LeWorldModel from converted shards
- `wally-train-curriculum` — train with progressive horizon curriculum (8 → 16 → 32 → full)
- `wally-plan` — plan action sequences using CEM-based MPC
- `wally-plan-hierarchical` — hierarchical planning with subgoal decomposition
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

## /opsx-apply workflow

When running `/opsx-apply`, delegate each task to a separate subagent via the Task tool instead of implementing tasks serially in the main conversation. This keeps each task focused and parallelizable.

Pattern for each task:
1. Read the task description and all relevant context files (specs, design, existing code)
2. Create a subagent via the Task tool with a detailed prompt covering: what to implement, which files to edit/create, how to verify (lint, typecheck, test commands), and relevant code conventions
3. The subagent returns when done — review the result, ensure tests pass, then mark the task complete in the tasks file
4. Move to the next task

Use the `general` subagent type for implementation tasks. Use the `explore` subagent type for research/investigation tasks.

Note: Use TDD (test-driven development) for tasks that contain complex logic, algorithms, or well-defined interfaces — write tests first to clarify requirements before implementation. 