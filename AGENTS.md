# AGENTS.md

## Project

Wally is a Minecraft AI research project. Goal: train world models (LeWorldModel-style) on collected gameplay trajectories, then use them for planning (CEM-based MPC) and goal-conditioned agents. Reference papers are in the repo root as PDFs.

## Hardware

- **GPU**: AMD Radeon RX 6700 XT (RDNA 2, GFX 10.3.0, 12.8GB VRAM)
- **GPU compute path**: **Windows-native ROCm via TheRock multi-arch PyTorch** (see `docs/gpu-setup.md`)
- **WSL2**: Used for data collection (`wally-collect`) and the photoreal MineStudio render in `wally-play --relay`. **WSL2 GPU compute is currently broken** — see `docs/gpu-setup.md#wsl2-compute-status-broken`.

## Two environments

Wally uses **two separate Python environments** depending on the task:

| Task | Environment | Reason |
|------|-------------|--------|
| Trajectory collection (`wally-collect`); photoreal agent loop with `wally-play --relay` | WSL2 (Podman container with `rocm/pytorch` image) | MineStudio's Java engine + LWJGL natives are Linux-only; the `wally-dev` container provides them and exposes the rendered POV over loopback to the Windows host |
| Training, planning, validation, deployment (`wally-train`, `wally-plan`, `wally-validate`, `wally-deploy`, `wally-convert`) | **Windows-native Python** with TheRock multi-arch PyTorch | librocdxg in WSL2 cannot submit compute commands to RDNA2 (gfx1031) hardware queues — see `docs/gpu-setup.md` |

## Setup commands

### Windows (training, planning, validation)

```powershell
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\Activate.ps1"
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -m pytest -m smoke -x --tb=short
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -m ruff check .
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -m mypy
```

### WSL2 (collector and `wally-play --relay`)

The collector uses `uv` inside the Podman container; the photoreal `wally-play --relay` workflow also runs inside the same `wally-dev` container. See `src/collector/AGENTS.md` for collector quirks and `docs/live-viewer.md#wally-play-in-wsl2` for the relay command.

## Project structure

Application code lives in `src/`:
- `src/collector/` — trajectory collection (MineStudio container only; see `src/collector/AGENTS.md`)
- `src/deployer/` — Minecraft server deployment (voxel renderer, action throttling, safety filters, ServerEnv adapter, CLI)
- `src/exporter/` — WebDataset shard export (`ShardWriter`, `generate_manifest`) — legacy, used by tests
- `src/validator/` — shard inspection and validation CLI (`inspect`, `validate`, `samples`)
- `src/wally/` — LeWorldModel training pipeline (encoder, predictor, planner, trainer; see `src/wally/AGENTS.md`)
- `src/agent/` — goal-conditioned agent loop, viewer, MJPEG POV relay, play CLI (see `src/agent/AGENTS.md`)
- `tools/` — standalone scripts (not entry points): `loss_dashboard.py`, `eval_goals.py`, `test_live_viewer.py`, plus ad-hoc shard inspection / repacking / verification scripts

Tests live in `tests/` covering all packages plus an end-to-end integration test.

## CLI entry points

- `wally-collect` — collect trajectories from Minecraft, saves raw `.tar` shards to `data/raw/`
- `wally-convert` — convert raw shards to training format (`.npz` per episode) in `data/shards/`
- `wally-train` — train LeWorldModel from converted shards
- `wally-train-curriculum` — train with progressive horizon curriculum (8 → 16 → 32 → full)
- `wally-plan` — plan action sequences using CEM-based MPC
- `wally-plan-hierarchical` — hierarchical planning with subgoal decomposition
- `wally-play` — run goal-conditioned agent loop locally via MineStudio; `--relay` exposes the POV over an MJPEG HTTP server (see `docs/live-viewer.md`)
- `wally-validate` — inspect/validate/sample shards
- `wally-deploy` — deploy trained agent to a Minecraft server (optional live OpenCV viewer via `--viewer {cv2,none}`)

## Live agent viewer

There are two production paths — pick based on where MineStudio can actually run. Details, end-to-end commands, and the WSL2 planner-performance caveat live in `docs/live-viewer.md`:

| Path | When to use | CLI |
|------|-------------|-----|
| `wally-deploy` against a local vanilla server | Windows + working Minecraft server on `localhost:25565` (voxel-grid `FrameRenderer`) | `wally-deploy --server localhost:25565 --checkpoint <ckpt> --goal-frame <goal.png> --viewer cv2` |
| `wally-play --relay` in WSL2, streamed to Windows | Photoreal MineStudio render only starts inside the WSL2 container; the agent loop runs there and the POV is streamed over an MJPEG HTTP relay at `http://localhost:8081/stream` | see `docs/live-viewer.md#wally-play-in-wsl2` |

## OpenSpec

All feature work goes through OpenSpec. Workflow: see `docs/openspec-workflow.md`.

## Code style

- Python 3.12+, follows PEP 8
- Match patterns from existing code
- `pyproject.toml` is the single source of truth for dependencies and tool config

## Testing instructions

- **Windows**: run smoke tests before every commit: `.\.venv-windows\Scripts\python.exe -m pytest -m smoke -x --tb=short`
- Add or update tests for any code you change
- The integration test (`tests/test_integration.py`) runs the full collect → convert → validate pipeline with a mock environment

### AG-Tests

- There are ag-tests to manual test the agent using codex/opencode


## External docs

This file is the slim root. Subpackage-specific rules apply on top and are auto-loaded via the directory walk:

- `src/agent/AGENTS.md` — agent loop, viewer, MJPEG relay
- `src/collector/AGENTS.md` — MineStudio container quirks
- `src/wally/AGENTS.md` — training, predictor, data format, checkpoint compat

All `docs/*.md` files are **on-demand** — read with the Read tool when a cross-reference in the files above is relevant to the task at hand. Do not preemptively load them.

- `docs/live-viewer.md` — two viewer paths, end-to-end commands, WSL2 planner perf
- `docs/openspec-workflow.md` — OpenSpec + `/opsx-apply` workflow
- `docs/gpu-setup.md` — Windows TheRock + WSL2 librocdxg + WSL2 compute status
