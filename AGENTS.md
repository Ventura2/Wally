# AGENTS.md

## Project

Wally is a Minecraft AI research project. Goal: train world models (LeWorldModel-style) on collected gameplay trajectories, then use them for planning (CEM-based MPC) and goal-conditioned agents. Reference papers are in the repo root as PDFs.

## Setup commands

- `uv sync` — install all dependencies
- `uv run pytest` — run the full test suite
- `uv run pytest tests/<file> -k "<name>"` — run a specific test
- `uv run ruff check .` — lint
- `uv run mypy` — typecheck

## Project structure

Application code lives in `src/` with three packages:
- `src/collector/` — trajectory collection (env wrapper, buffer, recorder, config, orchestrator)
- `src/exporter/` — WebDataset shard export (`ShardWriter`, `generate_manifest`)
- `src/validator/` — shard inspection and validation CLI (`validator.cli` with `inspect`, `validate`, `samples`)

Tests live in `tests/` covering all three packages plus an end-to-end integration test.

## Code style

- Python 3.12+, follows PEP 8
- Match patterns from existing code
- `pyproject.toml` is the single source of truth for dependencies and tool config

## Testing instructions

- Run the full suite with `uv run pytest` before every commit
- Add or update tests for any code you change
- The integration test (`tests/test_integration.py`) runs the full collect → export → validate pipeline with a mock environment

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
