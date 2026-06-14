# AGENTS.md

## Project

Wally is a Minecraft AI research project. Goal: train world models (LeWorldModel-style) on collected gameplay trajectories, then use them for planning (CEM-based MPC) and goal-conditioned agents. Reference papers are in the repo root as PDFs.

## Hardware

- **GPU**: AMD Radeon RX 6700 XT (RDNA 2, GFX 10.3.0, 12.8GB VRAM)
- **GPU compute path**: **Windows-native ROCm via TheRock multi-arch PyTorch** (see "GPU setup (Windows)" below)
- **WSL2**: Used for data collection (MineStudio) and librocdxg-based GPU detection. **WSL2 GPU compute is currently broken** — see "WSL2 compute status" below.

## Two environments

Wally uses **two separate Python environments** depending on the task:

| Task | Environment | Reason |
|------|-------------|--------|
| Trajectory collection (`wally-collect`) | WSL2 (Podman container with `rocm/pytorch` image) | MineStudio requires WSL2/Linux for the Minecraft Java engine |
| Training, planning, validation, deployment (`wally-train`, `wally-plan`, `wally-validate`, `wally-deploy`, `wally-convert`) | **Windows-native Python** with TheRock multi-arch PyTorch | librocdxg in WSL2 cannot submit compute commands to RDNA2 (gfx1031) hardware queues — see below |

## Setup commands

### Windows (training, planning, validation)

Use the dedicated Windows venv at `.venv-windows/`:

```powershell
# Activate venv
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\Activate.ps1"

# Run tests
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -m pytest -m smoke -x --tb=short

# Lint
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -m ruff check .

# Typecheck
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -m mypy
```

### WSL2 (collector only)

The collector uses `uv` inside the Podman container:

```sh
# Inside the container (wally-dev)
podman exec wally-dev sh -c 'cd /workspace && PYTHONPATH=src python3 -m wally.cli.collect --episodes 1 --output-dir data/raw --max-steps 5000'
```

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
- `tools/` — standalone scripts (not part of the installed entry points). Current contents:
  - `loss_dashboard.py` — tail a `wally-train --log-file` log, plot loss curves + ETA
  - `eval_goals.py` — load several checkpoints, plan toward long-horizon goals (get_wood, get_iron_ore, get_stone, navigate_look_around), report success/latent-distance per (checkpoint, goal) across `world_model` / `minestudio` / `mock` backends
  - `*` — shard inspection / repacking / verification scripts (see `tools/audit_diamond.py`, `tools/verify_shard.py`, etc. for examples)

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

## Standalone tools (not installed as entry points)

Run with the venv's Python directly:

- `python tools/loss_dashboard.py --log-file <path> [--config <yaml>] [--output <png>] [--live]` — live loss-curve + ETA dashboard for a `wally-train` run.
- `python tools/eval_goals.py --checkpoints '<glob>' --mode {world_model,minestudio,mock} [--config <yaml>] [--num-checkpoints N] [--episodes N] --output <dir>` — per-checkpoint long-horizon goal eval (success rate, latent distance, plan cost). See `--help` for goal list and CEM tuning.

## GPU setup (Windows) — recommended for training

Use **TheRock multi-arch PyTorch** nightly wheels with `device-gfx1031` extra. This installs PyTorch with the AMD ROCm runtime and gfx1031-specific (RX 6700 XT) kernel packs, using AMD's official Adrenalin driver D3D12 compute path on Windows directly.

### Prerequisites

- **Windows**: AMD Adrenalin driver (any recent version that ships with the D3D12 driver)
- **Python**: 3.12 or 3.13 (3.14 not yet supported by TheRock wheels as of June 2026)
- **No WSL2 librocdxg needed** for training on Windows

### Install TheRock PyTorch

```powershell
# Create venv
python -m venv .venv-windows

# Upgrade pip
.\.venv-windows\Scripts\python.exe -m pip install --upgrade pip

# Install TheRock multi-arch PyTorch with gfx1031 (RX 6700 XT) kernel packs
.\.venv-windows\Scripts\python.exe -m pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx1031]"

# Install torchvision from the same index (otherwise pip installs the CPU/wrong-ABI wheel)
.\.venv-windows\Scripts\python.exe -m pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ --force-reinstall --no-deps torchvision

# Install the wally project in editable mode (this will also pull numpy, etc.)
.\.venv-windows\Scripts\python.exe -m pip install -e .
```

Total download: ~1GB (rocm-sdk-core is 745MB, rocm-sdk-libraries 116MB, amd-torch-device-gfx1031 45MB).

### Verify GPU compute

```powershell
.\.venv-windows\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True AMD Radeon RX 6700 XT
```

Quick perf check (8K matmul should reach ~10 TFLOPS FP32):

```python
import torch, time
a = torch.randn(8192, 8192, device='cuda'); b = torch.randn(8192, 8192, device='cuda')
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10): c = a @ b
torch.cuda.synchronize()
print(f'{10/(time.time()-t0):.1f} matmul/s')
```

### Known issues (Windows)

- MIOpen warning: `CK grouped conv library not found for device gfx1031: No se puede encontrar el m�dulo especificado.` — benign, falls back to a non-CK kernel path.
- Pip may downgrade numpy to 1.26.4 when installing wally (some transitive dep). This is fine — PyTorch handles both numpy 1.x and 2.x.

## GPU setup (WSL2) — for collector only

The RX 6700 XT (RDNA2) is **not** in AMD's official WSL2 ROCm compatibility matrix (which only lists RDNA3/RDNA4 and Ryzen AI APUs). However, GPU **detection** works with a custom `dids.conf` entry. The setup requires building AMD's open-source `librocdxg` library from source.

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

### WSL2 compute status: BROKEN

**librocdxg v1.2.0 cannot submit compute commands to RDNA2 (gfx1031) hardware queues in WSL2.** The D3DKMT command submission to the GPU's hardware queue processor never completes — the GPU never executes the command, so fences are never signaled, and HIP waits forever.

**What works in WSL2:**
- `rocminfo` enumerates GPU correctly (gfx1031, 12.8GB VRAM)
- `hsa_init` succeeds, GPU agent detected
- `hipInit(0)` returns success
- `hipMalloc` / `hipFree` (GPU memory allocation)
- `torch.cuda.is_available()` = True
- `torch.empty(...)` (memory allocation)

**What hangs (librocdxg SDMA submission failure):**
- `hipMemcpy` (any direction)
- Any HIP kernel launch
- `torch.zeros`, `tensor.to('cuda')` (PyTorch ops)
- Custom HIP kernels compiled with hipcc

**Diagnostic log from `AMD_LOG_LEVEL=4`:**
```
:3:rocdevice.cpp :2871: Number of allocated hardware queues with low priority: 0, with normal priority: 0, with high priority: 0
:3:rocdevice.cpp :2952: Created SWq=0x... to map on HWq=0x...
:4:rocdevice.cpp :2019: Allocate hsa host memory 0x..., size 0x400000
<HANGS HERE — D3DKMT command submission>
```

The HWq object is created in userspace but the D3DKMT command submission to the GPU's hardware queue processor never completes. This is a librocdxg limitation, not something fixable from the user side. No known GitHub issues for "command queue hang" or "RDNA2 compute" in librocdxg.

**Do not attempt training, planning, or any compute workload in WSL2.** Use the Windows-native TheRock setup instead.

### Known issues (WSL2, collector only)

- `rocm-smi` is a Python script and may fail if `python3` isn't on PATH in the shell
- `dmesg` will show `dxgkio_query_adapter_info: Ioctl failed: -22` — this is from the amdgpu kernel module and is **benign** (userspace path works fine)
- `rocminfo` shows `Warning: Windows driver is old` — this is a non-fatal warning
- Adrenalin 26.6.1 driver triggers this warning; older or newer drivers may or may not

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

## Checkpoint compatibility

Pre-AdaLN checkpoints (saved before the `lewm-adaln-predictor` change) use a different model architecture (interleaved-input TransformerEncoder) and cannot be loaded by the current code. They are archived in `checkpoints/_incompatible_pre_adaln/`. New training runs start from step 0.

## Predictor architecture

The LeWorldModel predictor uses the official LeWM AdaLN-Zero design (`lucas-maes/le-wm/module.py`). Actions are passed as a conditioning sequence `c` to the `Transformer` (NOT interleaved into the latent sequence). Internal LayerNorms are `nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)` — no learnable `weight` parameter, so bf16 gradients cannot overflow. The AdaLN-Zero modulation linear (`self.modulation = nn.Linear(c_dim, 6*dim)`) is zero-initialized, so every `ConditionalBlock` is a strict identity at step 0. The previous interleaved-input design (and its `autocast(enabled=False)` fp32 wrapper) was the root cause of the bf16 NaN-gradient bug and has been removed.

## SIGReg alpha

Default `alpha: 0.1` in `configs/lewm_default.yaml` matches the LeWM paper Section 3.1 (Algorithm 1). The previous `0.01` sat at the lower edge of the paper's safe range. SIGReg is applied to the **projected** encoder output (the output of the `projector` MLP, not the raw encoder output), matching `lucas-maes/le-wm/jepa.py:39`.

## Code style

- Python 3.12+, follows PEP 8
- Match patterns from existing code
- `pyproject.toml` is the single source of truth for dependencies and tool config

## Testing instructions

- **Windows**: run smoke tests before every commit: `.\.venv-windows\Scripts\python.exe -m pytest -m smoke -x --tb=short`
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