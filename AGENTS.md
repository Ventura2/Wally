# AGENTS.md

## Project

Wally is a Minecraft AI research project. Goal: train world models (LeWorldModel-style) on collected gameplay trajectories, then use them for planning (CEM-based MPC) and goal-conditioned agents. Reference papers are in the repo root as PDFs.

## Hardware

- **GPU**: AMD Radeon RX 6700 XT (RDNA 2, GFX 10.3.0, 12.8GB VRAM)
- **GPU compute path**: **Windows-native ROCm via TheRock multi-arch PyTorch** (see `docs/gpu-setup.md`)
- **WSL2**: Used for data collection (`wally-collect`) and the photoreal MineStudio render in `wally-play --relay`. **WSL2 GPU compute is currently broken** — see `docs/gpu-setup.md#wsl2-compute-status-broken`.

## Training requires a GPU

`wally-train`, `wally-train-curriculum`, and `wally-train-hierarchy` always
train on GPU. The CLIs default to `--device cuda` and exit with a clear
error if `torch.cuda.is_available()` is False (typically because the active
venv has a CPU-only torch build — reinstall from the TheRock multi-arch
index per `docs/gpu-setup.md`). CPU is exposed only as an explicit
`--device cpu` escape hatch used by a handful of fast smoke tests on tiny
configs; it emits a warning. Never run a real training job on CPU — there
is no auto-fallback and no environment variable that re-enables it.

## Two environments

Wally uses **two separate Python environments** depending on the task:

| Task | Environment | Reason |
|------|-------------|--------|
| Trajectory collection (`wally-collect`); photoreal agent loop with `wally-play --relay` | WSL2 (Podman container with `rocm/pytorch` image) | MineStudio's Java engine + LWJGL natives are Linux-only; the `wally-dev` container provides them and exposes the rendered POV over loopback to the Windows host |
| Training, planning, validation, deployment (`wally-train`, `wally-train-hierarchy`, `wally-plan`, `wally-validate`, `wally-deploy`, `wally-convert`) | **Windows-native Python** with TheRock multi-arch PyTorch | librocdxg in WSL2 cannot submit compute commands to RDNA2 (gfx1031) hardware queues — see `docs/gpu-setup.md` |

## Quick start: train a small wood model and run the agent

This is the exact sequence that produced `checkpoints/wood_1000/checkpoint_1000.pt`
and the E2E test in `ag-tests/run_wood_1k/episode_0.npz`. Use it as a
template for any small smoke-test run.

**Assumptions:** wood data already in `data/shards/treechop_full/`
(see "Training workflow" below if you need to (re-)convert), checkpoint
output goes to a fresh directory under `checkpoints/`.

### 1. Train (Windows, GPU, ~10 min for 1000 steps)

```powershell
# Copy an existing config and edit output_dir / max_steps as needed
Copy-Item configs\lewm_wood_500.yaml configs\lewm_wood_<N>.yaml

# Run from the project root, venv activated
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" `
    -m wally.cli.train `
    --config configs/lewm_wood_<N>.yaml `
    --log-file logs/wood_train_<N>.log
```

The trainer logs `fetch=Xs gpu=Ys total=Zs` per step (`src/wally/training/trainer.py:245-263`).
- Expected on the chunked data: `gpu ≈ 0.05-0.07s` per step, `total ≈ 0.5-0.8s` (fetch dominates on first batch, then the prefetcher keeps the queue full)
- 100 steps ≈ 1 min, 1000 steps ≈ 8-10 min, 5000 steps ≈ 45-60 min
- Checkpoints every `checkpoint_interval` steps → `checkpoints/<output_dir>/checkpoint_<step>.pt` (~45 MB each)

### 2. Run the agent in the WSL2 container with the MJPEG relay

```powershell
# 2a. Start the podman machine if it's not running
podman machine start

# 2b. Start (or create) the wally-dev container with port 8081 mapped
podman start wally-dev 2>$null
if ($LASTEXITCODE -ne 0) {
    podman run -d --name wally-dev --hostname wally-dev --network pasta `
        -v D:\Projects\Personal\artificial-intelligence\wally:/workspace:rbind `
        -p 8081:8081 `
        localhost/wally-dev:latest sleep infinity
}

# 2c. Write a start script and copy it into the container
$script = @'
#!/bin/bash
export PYTHONPATH=/workspace/src
export MINESTUDIO_DIR=/tmp/MineStudio
exec python3 -m wally.agent.play \
  --relay --relay-host 0.0.0.0 --relay-port 8081 \
  --checkpoint /workspace/checkpoints/<output_dir>/checkpoint_<N>.pt \
  --goal-frame /workspace/checkpoints/goal_frame1.png \
  --planner cem --viewer none \
  --config /workspace/checkpoints/ag_test_wood.yaml \
  --record --output-dir /workspace/ag-tests/run_<name>
'@
Set-Content logs\start-play-<name>.sh -Value $script -NoNewline
podman cp logs\start-play-<name>.sh wally-dev:/tmp/start-play.sh
podman exec wally-dev chmod +x /tmp/start-play.sh
podman exec wally-dev mkdir -p /workspace/ag-tests/run_<name>

# 2d. Start detached
podman exec -d wally-dev bash -c 'setsid nohup /tmp/start-play.sh > /tmp/wally-play.log 2>&1 < /dev/null & disown'

# 2e. Verify the relay is up
Start-Sleep 12
podman exec wally-dev curl -s -m 3 http://localhost:8081/healthz   # -> "ok"
```

Open `http://localhost:8081/stream` in any browser to watch the agent.
Episode finishes at `episode_timeout` (default 1000 steps, ~110s; use a
shorter config to iterate faster — see `configs/ag_test_wood.yaml` style
files in `checkpoints/`).

### 3. Analyze the trajectory

```powershell
# Copy the trajectory out of the container
podman cp wally-dev:/workspace/ag-tests/run_<name>/episode_0.npz ag-tests\run_<name>\episode_0.npz

# Run the analyzer (verdict at the bottom: SUCCESS / FAIL with reasons)
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" `
    tools\analyze_trajectory.py ag-tests\run_<name>\episode_0.npz
```

The analyzer reports per-action stats, inventory changes, wood-related
events, and a final verdict. If the FAIL line says "CEM local minimum
detected" with > 50 inventory-spam steps, the agent got stuck opening
the inventory — either train more steps or apply the `action[12]=0`
mask in `src/wally/agent/loop.py` (see `src/wally/agent/AGENTS.md`).

### Expected results by training size (wood model, vit_tiny, depth=4)

| Training steps | Wall time | Final cost | Behavior observed |
|----------------|-----------|-----------|-------------------|
| 100 | 1 min | ~100k | Random shake + inventory loop, no wood |
| 1,000 | 8-10 min | ~2k | Walks toward trees, no inventory loop, still no wood |
| 5,000 | 45-60 min | (untested) | Likely walks up to a tree and chops |
| 10,000 | ~30 min (optimized) | ~12k | Walks in straight lines, swings at whatever is in front. With a tree-frame `g1` (see `logs/make_g1_tree.py`) the agent breaks 600+ blocks/episode — all `tall_seagrass`, not wood. |

The 1k→10k step is the one that has the biggest user-visible effect
on motor quality (cleaner walks, more coordinated attack patterns), but
**the L0+L1 stack with K=4 horizons cannot plan a 50-100 step
"approach tree → face trunk → chop → pick up" sequence** — that is
the L2's job. See `src/wally/hierarchy/AGENTS.md#l2-path-is-not-viable-yet`
for why L2 isn't trained end-to-end yet and what it would take.

### Dataclass / dataloader perf tuning

L0 training with the default dataloader config (`num_workers=4`) runs
at ~0.45 s/step on the chunked shards (14% GPU utilization — the
dataloader is the bottleneck). The current configs
(`configs/lewm_wood_10000.yaml`, `configs/hierarchy_l1_5k.yaml`) set
`num_workers: 8, persistent_workers: true, prefetch_factor: 4`, which
takes the L0 to ~0.18 s/step (GPU-bound). If you copy a config and
see long `fetch=` times, you're dataloader-bound; bump those three
settings.

## Setup commands

### Windows (training, planning, validation)

```powershell
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\Activate.ps1"
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -m pytest -m smoke -x --tb=short
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -m ruff check .
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -m mypy
```

### Training workflow (collect -> convert -> train)

The CLIs form a three-step pipeline. Each step's output is the next step's input.

```powershell
# 1. Collect (WSL2, MineStudio container) -> data/raw/<task>/shard_*.tar
wally-collect ...

# 2. Convert (Windows, TheRock venv) -> data/shards/<task>/shard_*.tar
#    Each raw .tar becomes a converted .tar with one .npz per episode
#    (frames: uint8 (T,224,224,3), actions: float32 (T,25)).
& .venv-windows\Scripts\python.exe -m wally.cli.convert `
    --input data/raw/<task> `
    --output data/shards/<task> `
    --config configs/converter_default.yaml

# 3. Train (Windows, TheRock venv, GPU) -> checkpoints/<run>/checkpoint_<step>.pt
#    `data_dir` in the YAML config is the ONLY way to select which shards
#    to train on - there is no --data-dir CLI flag, no task filter, no
#    glob. Just point data_dir at the converted shard directory.
& .venv-windows\Scripts\python.exe -m wally.cli.train `
    --config configs/<your_config>.yaml `
    --log-file logs/<run>.log
```

See `docs/gpu-setup.md#how-to-launch-training-on-gpu` for the end-to-end
launch checklist (venv activation, PATH for ROCm DLLs, PYTHONPATH for the
src layout, AV exceptions, perf tuning, monitoring).

### WSL2 (collector and `wally-play --relay`)

The collector uses `uv` inside the Podman container; the photoreal `wally-play --relay` workflow also runs inside the same `wally-dev` container. See `src/wally/collector/AGENTS.md` for collector quirks and `docs/live-viewer.md#wally-play-in-wsl2` for the relay command.

## Project structure

Application code lives under `src/wally/`. Subpackages:
- `src/wally/collector/` — trajectory collection (MineStudio container only; see `src/wally/collector/AGENTS.md`)
- `src/wally/deployer/` — Minecraft server deployment (voxel renderer, action throttling, safety filters, ServerEnv adapter, CLI)
- `src/wally/exporter/` — WebDataset shard export (`ShardWriter`, `generate_manifest`) — legacy, used by tests
- `src/wally/validator/` — shard inspection and validation CLI (`inspect`, `validate`, `samples`)
- `src/wally/training/`, `src/wally/models/`, `src/wally/planner/`, `src/wally/data/`, `src/wally/config/`, `src/wally/cli/` — LeWorldModel training pipeline (see `src/wally/AGENTS.md`)
- `src/wally/agent/` — goal-conditioned agent loop, viewer, MJPEG POV relay, play CLI (see `src/wally/agent/AGENTS.md`)
- `src/wally/hierarchy/` — L1/L2/L3 JEPA world-model stack on top of the frozen L0 LeWorldModel, continuous-embedding message bus, drift detection, hierarchical-embedding planner, training loop (see `src/wally/hierarchy/AGENTS.md`)
- `tools/` — standalone scripts (not entry points): `loss_dashboard.py`, `eval_goals.py`, `test_live_viewer.py`, plus ad-hoc shard inspection / repacking / verification scripts

Tests live in `tests/` covering all packages plus an end-to-end integration test.

## CLI entry points

- `wally-collect` — collect trajectories from Minecraft, saves raw `.tar` shards to `data/raw/`
- `wally-convert` — convert raw shards to training format (`.npz` per episode) in `data/shards/`
- `wally-train` — train LeWorldModel from converted shards
- `wally-train-curriculum` — train with progressive horizon curriculum (8 → 16 → 32 → full)
- `wally-train-hierarchy` — train L1/L2/L3 JEPA world-model layers on top of a frozen L0 checkpoint (see `src/wally/hierarchy/AGENTS.md` and `docs/hierarchical-world-model.md`)
- `wally-plan` — plan action sequences using CEM-based MPC
- `wally-plan-hierarchical` — hierarchical planning with subgoal decomposition
- `wally-play` — run goal-conditioned agent loop locally via MineStudio; `--relay` exposes the POV over an MJPEG HTTP server (see `docs/live-viewer.md`). Supports `--planner hierarchical-embedding --hierarchy-checkpoint <ckpt> --target-embedding <g3.pt>` for the multi-layer JEPA planner
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

- `src/wally/agent/AGENTS.md` — agent loop, viewer, MJPEG relay
- `src/wally/collector/AGENTS.md` — MineStudio container quirks
- `src/wally/AGENTS.md` — training, predictor, data format, checkpoint compat
- `src/wally/hierarchy/AGENTS.md` — L1/L2/L3 JEPA world-model stack, frozen-L0 invariant, training order

All `docs/*.md` files are **on-demand** — read with the Read tool when a cross-reference in the files above is relevant to the task at hand. Do not preemptively load them.

- `docs/live-viewer.md` — two viewer paths, end-to-end commands, WSL2 planner perf
- `docs/openspec-workflow.md` — OpenSpec + `/opsx-apply` workflow
- `docs/gpu-setup.md` — Windows TheRock + WSL2 librocdxg + WSL2 compute status
- `docs/hierarchical-world-model.md` — L1/L2/L3 training commands, variable-depth runtime, drift-threshold tuning
