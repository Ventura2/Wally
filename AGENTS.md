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
- **The 5k config (`configs/lewm_wood_5000.yaml`) has `early_stop: true` enabled** so the actual wall time is ~25 min and training stops at the loss plateau (see "Early stopping" below). The final `checkpoint_best.pt` has the lowest-EMA-loss weights.
- Checkpoints every `checkpoint_interval` steps → `checkpoints/<output_dir>/checkpoint_<step>.pt` (~45 MB each). With `early_stop: true`, also `checkpoint_best.pt` is saved whenever the EMA of `total_loss` improves.

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
| 5,000 (raw, `early_stop: false`) | 45-60 min | ~2.5k | **Loss plateaus at step ~1500**; the remaining 3.5k steps are wasted. Same quality as `early_stop: true` but spends the wasted compute. |
| 5,000 config (`early_stop: true`, effective ~1.5-3k) | **~25 min** | ~2.5k | Same quality L0 as raw 5k. `checkpoint_best.pt` is the best weights — use this, not the final-step checkpoint. |
| 10,000 | ~30 min (optimized) | ~12k | Walks in straight lines, swings at whatever is in front. With a tree-frame `g1` (see `logs/make_g1_tree.py`) the agent breaks 600+ blocks/episode — all `tall_seagrass`, not wood. |

**The 1k→10k wall-time difference is mostly wasted compute, not learning.** Both 5k and 12k runs plateau at step ~1500 in `total_loss` (see `logs/wood_train_5k.log` and `logs/wood_train_12000.log`). The plateau is a *representational* attractor: the L0 has learned the easiest solution that fits the data ("match the goal's brightness") and additional steps just oscillate around it. With `early_stop: true`, future training runs automatically stop at this plateau and save `checkpoint_best.pt` — there's no behavioral penalty for stopping early.

The 1k→10k step *does* have a user-visible effect on motor quality (cleaner walks, more coordinated attack patterns), but **the L0+L1 stack with K=4 horizons cannot plan a 50-100 step "approach tree → face trunk → chop → pick up" sequence** — that is the L2's job. See `src/wally/hierarchy/AGENTS.md#l2-path-is-not-viable-yet` for why L2 isn't trained end-to-end yet and what it would take.

### Dataclass / dataloader perf tuning

L0 training with the default dataloader config (`num_workers=4`) runs
at ~0.45 s/step on the chunked shards (14% GPU utilization — the
dataloader is the bottleneck). The current configs
(`configs/lewm_wood_10000.yaml`, `configs/hierarchy_l1_5k.yaml`) set
`num_workers: 8, persistent_workers: true, prefetch_factor: 4`, which
takes the L0 to ~0.18 s/step (GPU-bound). If you copy a config and
see long `fetch=` times, you're dataloader-bound; bump those three
settings.

## Early stopping

**The wood L0 plateau is at step ~1500** (see "Expected results by
training size" above). The trainer has an EMA-based early stop that
detects this automatically. Configured in the YAML under `training:`:

```yaml
training:
  early_stop: true               # default: false
  early_stop_patience: 500       # stop if EMA total_loss doesn't improve for N steps
  early_stop_min_step: 1000      # don't consider stopping before this step
  early_stop_ema_alpha: 0.1      # EMA smoothing factor (lower = smoother)
  early_stop_min_delta: 0.0      # minimum improvement to count as "better"
```

When the EMA of `total_loss` doesn't improve for `patience` steps
(after `min_step`), training stops and `checkpoint_best.pt` holds the
lowest-EMA-loss weights. The 5k config defaults to
`patience=500, min_step=1000`, which means the 5k run that previously
took 48 min now stops around step 2500-3000 (~25 min) with the same
quality L0. For 12k runs, the same plateau means early stop saves ~75%
of the training time.

**Validation log excerpt** (from a 600-step smoke test,
`lewm_wood_earlystop_test.yaml`, `patience=100, min_step=200`):
```
Step 200 | New best EMA total_loss=0.2107 (saved checkpoint_best.pt)
Step 396 | New best EMA total_loss=0.1361 (saved checkpoint_best.pt)
Step 496 | Early-stop trigger: 100 steps since best (patience=100).
         | Best EMA total_loss=0.1361 at step 396.
Step 496 | Training stopped early ... Use checkpoint_best.pt for the best weights.
```

**When to enable early stop (default for new L0 runs):**
- Iterating on data or architecture — saves 50-75% of GPU time per run
- CI / smoke tests — let patience decide instead of guessing `max_steps`
- Any run where you only need "good enough" — the plateau is the same regardless of total step count

**When to disable early stop:**
- Researching learning-rate schedules (you want to see the full cosine decay)
- Reproducing a known-good long run (use `max_steps` instead)
- Comparing schedules — early stop masks differences in convergence rate

**Important:** early stop is *orthogonal* to the architectural problem.
The plateau is a *representational* attractor (the 1-D brightness meter
— see `tools/experiments/REPORT.md` §D for the PCA probe), not a
training-time one. Stopping earlier doesn't break the L0 out of this
basin. To break out you need an architecture change (FF-JEPA
hierarchy, see `src/wally/hierarchy/AGENTS.md`) or a different
latent-objective (whitening, contrastive, VICReg). The early stop
saves compute; the ceiling is in the architecture, not the training
budget.

## Latent collapse (VICReg)

The L0 plateau is *representational*: PCA on the projected 192-dim
latent shows PC1 ≈ 84% of variance and `‖z‖` correlates with frame
brightness at r ≈ +0.97. The JEPA prediction loss rewards "predict
the next z well", and brightness is a strong predictor of next-frame
brightness, so the L0 collapses to a 1-D brightness meter. SIGReg
(`alpha: 0.1`) is too soft to break this shortcut on its own.

VICReg (Bardes et al. 2022) adds two auxiliary loss terms to the
projected encoder output that directly attack the collapse: a hinge
on per-dim std (`mean(relu(gamma - z.std(dim=0)))`) and a squared
off-diagonal covariance penalty. Three new YAML knobs control it
alongside `alpha` and `early_stop`:

```yaml
training:
  vicreg_weight: 1.0       # gate (0 disables; default in lewm_default.yaml is 1.0)
  vicreg_std_target: 1.0   # hinge gamma
  vicreg_cov_weight: 1.0   # weight of the off-diag penalty
```

When `vicreg_weight = 0` the VICReg term is skipped entirely (no
extra `std`/`cov` allocations) and the metrics dict stays
bit-identical to the pre-VICReg output. Per-run `lewm_wood_*.yaml`
configs intentionally do *not* set these fields — the per-run
baselines stay on the pre-change behavior. Full design, migration
plan, and post-merge retrain/eval tasks live in
`openspec/changes/l0-vicreg-decollapse/`.

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
- `tools/` — standalone scripts (not entry points): `loss_dashboard.py`, `eval_goals.py`, `test_live_viewer.py`, plus ad-hoc shard inspection / repacking / verification scripts. For agent-run debugging, `extract_anomalies.py` produces a single labeled contact sheet (PNG + JSON) of the most interesting moments in an `episode_0.npz` (inv-spam, camera-shake, cost-spike, attack-burst, first-event, brightness extremes, best-match-to-goal, final-frame). This is the offline counterpart to the relay stream: instead of 5 evenly-spaced frames (`extract_frames.py` — useful for a quick "what did the run look like" glance), you get 8 panels of 5-frame windows around the actual anomalies so an LLM can see *what went wrong*.

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


## Debugging complex bugs: the replicate-first workflow

For bugs that span multiple files or involve the interaction between modules (camera, inventory, action schema, planner/loop/env contracts, anything where the wrong index or wrong scale silently corrupts the agent's behavior), follow this workflow. **Never guess the fix and never apply it without first pinning the bug down with a test.**

### The four-step pattern

1. **Hypothesize** — read the relevant code in parallel and identify which line(s) could be wrong
2. **Replicate** — write a unit test that **fails on the current code** and **passes on the fix**
3. **Fix minimally** — change the smallest amount of code to make the test pass
4. **Verify end-to-end** — re-run the agent in WSL2 and compare the new trajectory against a saved baseline

### Step 1: Hypothesize

Read the touchpoint files in parallel. For action-schema bugs the touchpoints are always the same:

| File | What it does |
|---|---|
| `src/wally/agent/loop.py` | Per-step action mutation: clamps, masks (e.g. `action[12]=0`), EMA smoothing |
| `src/wally/agent/env.py` | Agent vocab → MineStudio action dict translation (camera rescale, key mapping) |
| `src/wally/agent/planner_factory.py` | Planner construction |
| `src/wally/planner/rollout.py` | `_translate_agent_action_to_l0` (agent vocab → training vocab permutation) |
| `src/wally/planner/actions.py` | `MineStudioActionVocab` — the source of truth for the index layout |
| `src/wally/data/dataset.py` | Training-time action clamping (`actions.clamp(-1, 1)`) |
| `src/wally/collector/env.py` | Collector-side: MineStudio returns raw degrees |

For each file, ask: "what scale is this value in?" If two files disagree on the scale (e.g. one thinks camera is in degrees, another thinks it's in [-1, 1] normalized), that's the bug.

### Step 2: Replicate

Create `tests/test_<bug_name>_replication.py`. The test file has two kinds of tests:

- **Contract tests** — pin down the index layout, vocab names, etc. These should pass on both the broken and fixed code. They exist so the index assignments can't silently change.
- **Bug tests** — assert the correct behavior. These should **fail on broken code with a specific error** and **pass on fixed code**.

Run the tests and read the exact failure messages — they tell you which line is wrong. If a test fails with a generic error (e.g. shape mismatch with no clear cause), the test is not specific enough; rewrite it to assert a value, not a shape.

```powershell
# Write the test file, then:
& ".venv-windows\Scripts\python.exe" -m pytest tests/test_<bug>_replication.py -v --tb=short
# Confirm: contract tests pass, bug tests fail with specific errors
```

### Step 3: Fix minimally

Change the smallest amount of code to make the failing tests pass. Do not refactor surrounding code. If a comment is now wrong (e.g. "the env rescales by 180" when it doesn't), update the comment in the same edit.

### Step 4: Verify end-to-end

Agent bugs only matter if they change behavior in the real env. Re-run the agent with the same checkpoint + goal as a previous run, save the new trajectory to a new dir, and compare:

```powershell
# Start the new run (see "2. Run the agent" above for the full command)
# Use a fresh output dir, e.g. ag-tests/run_<name>_fixed/

# After the episode, copy the trajectory out and run the analyzer
& ".venv-windows\Scripts\python.exe" tools\analyze_trajectory.py ag-tests\run_<name>_fixed\episode_0.npz

# For a side-by-side comparison, write tools/compare_<fix>.py that loads both
# trajectories and prints the key metrics (camera std, attack count,
# brightness end, cost progression, SCSA Spearman)
```

Metrics that should improve after fixing an action-schema bug:

| Metric | What to check |
|---|---|
| `camera_pitch` / `camera_yaw` std | Should be < 0.1 (bounded by the clamp) |
| `attack > 0.5` count | Should be > 0 (was being clamped to 0 by the bug) |
| Brightness end | Should be in the same range as the start (no drift to > 100 = sky) |
| Cost progression | Should be reducing (negative trend), not increasing |
| SCSA Spearman | Should be > 0.5 for most replans |
| Inventory action | Should always be 0 (the CEM local-minimum mask is active) |

### Why this works

- **Replicating first** gives ground truth — you know exactly what the bug is, not what you think it is
- **Tests pin the fix** so the bug can't silently regress in a future refactor
- **End-to-end verification** confirms the fix actually changes agent behavior, not just makes a test pass
- **Comparison script** becomes a regression baseline for future runs

### Example: the camera bug (2026-06-28)

Two bugs were found and fixed in one session:

1. `src/wally/agent/loop.py` clamped `action[10:11]` (attack/drop) instead of `action[0:1]` (camera) — the clamp was written against the **training** schema instead of the **agent** schema
2. `src/wally/agent/env.py` applied `* 180.0` to the camera, but the L0 was trained on raw degrees clamped to `[-1, 1]`, so the env was 180× over-rescaling

The fix was pinned by 19 tests in `tests/test_camera_bug_replication.py`. End-to-end comparison (`tools/compare_camera_fix.py`) showed the agent went from "always looking at sky, never attacks" to "stable camera, attacks 26% of steps, cost -42%". Before/after: `ag-tests/run_wood_5k_trm/episode_0.npz` vs `ag-tests/run_wood_5k_trm_fixed/episode_0.npz`.

### Example: the inventory mask

The loop has `action[12] = 0.0` to force inventory=0 (CEM local-minimum workaround — see `src/wally/agent/AGENTS.md`). A test `test_loop_does_not_mutate_inventory_action` was asserting the opposite (that inventory=1.0 passes through). The test was renamed to `test_loop_masks_inventory_action_to_zero` and flipped to assert `== 0.0` with a comment explaining the workaround. The fix is one line; the regression test is the durable part.

### Checklist

- [ ] Read all touchpoint files in parallel
- [ ] Identify the exact line(s) that could be wrong
- [ ] Write `tests/test_<bug>_replication.py` with contract + bug tests
- [ ] Run tests; confirm bug tests fail with **specific** errors
- [ ] Apply minimal fix (no surrounding refactors)
- [ ] Re-run tests; all pass
- [ ] Run `pytest -m smoke`; no regressions
- [ ] Run `ruff check`; no new warnings
- [ ] Re-run agent end-to-end in WSL2 with a fresh output dir
- [ ] Compare against saved baseline (camera std, attack count, brightness, cost, SCSA)
- [ ] Update `AGENTS.md` if the fix changes user-visible behavior
- [ ] Save a `tools/compare_<fix>.py` for future regression checks

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
