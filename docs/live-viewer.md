# Live agent viewer

The agent's POV frame is exposed via `info["pov"]` on every `env.step()` and rendered by a `FrameViewer` (local OpenCV window) or an MJPEG HTTP relay (for cross-environment viewing). Two production paths exist — pick based on where MineStudio can actually run.

## Paths

| Path | When to use | CLI |
|------|-------------|-----|
| **`wally-deploy` against a local vanilla server** | Windows has a working Minecraft server (1.18.1 recommended) reachable on `localhost:25565`. The `deployer.ServerEnv` uses pyCraft + a `FrameRenderer` voxel ray-cast to walk the real world. | `wally-deploy --server localhost:25565 --checkpoint <ckpt> --goal-frame <goal.png> --viewer cv2` |
| **`wally-play` from inside the WSL2 `wally-dev` container, streamed to Windows** | MineStudio's `runtime/` + `mcprec-6.13.jar` are Linux-only; the photoreal MineStudio render only starts inside the WSL2 container. The agent loop runs in the container and the latest POV is streamed over an MJPEG HTTP relay at `http://localhost:8081/stream` — open that URL in any browser or `cv2.VideoCapture` client on Windows. See "wally-play in WSL2" below. | inside container: `PYTHONPATH=src python3 -m wally.agent.play --relay --checkpoint /workspace/checkpoints/<ckpt>.pt --goal-frame /workspace/data/goal.png --planner cem --viewer none` ; on Windows: open `http://localhost:8081/stream` |

## wally-deploy (Windows, vanilla server)

The voxel-grid renderer's frames are stylized (no textures, no lighting) but reflect the actual chunk topology the server sends. For the photoreal MineStudio render, follow the **`wally-play` in WSL2** path below.

```powershell
# Terminal 1: start the local 1.18.1 server (symlinked from minecraft-server/)
cd D:\Projects\Personal\artificial-intelligence\wally
java -Xmx2G -jar minecraft-server\server.jar nogui

# Terminal 2: run the agent with a live window
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\Activate.ps1"
wally-deploy --server localhost:25565 `
  --checkpoint checkpoints\checkpoint_100000.pt `
  --goal-frame checkpoints\goal_frame1.png `
  --viewer cv2
```

Press `q` or `Esc` in the OpenCV window to stop the episode cleanly.
Use `--viewer none` (or `--no-viewer`) to disable the window for headless runs and CI.

## wally-play in WSL2

The `wally-play` CLI runs the goal-conditioned agent loop against the photoreal MineStudio render. MineStudio's Java engine + LWJGL natives only run on Linux, so the loop must execute inside the `wally-dev` Podman container. The container exposes the latest POV frame over an MJPEG HTTP relay so any OpenCV client on the Windows host (browser, `cv2.VideoCapture("http://localhost:8081/stream")`, VLC) can watch it.

### Verified end-to-end run (Windows host + Podman container)

This is the working sequence — copy-pasteable. Assumes the project is at `D:\Projects\Personal\artificial-intelligence\wally` on Windows (mounted at `/workspace` inside the container), and that a checkpoint + goal frame already exist in `checkpoints/`.

**Caveat:** the command below assumes the `wally-dev` container was already
created with the relay port mapped (`-p 8081:8081`). If you need to create
it fresh, the first `podman start` will fail and you need the
`podman run -d ...` line. The wrapper PowerShell in the root `AGENTS.md`
("Run the agent in the WSL2 container with the MJPEG relay" section)
handles both cases automatically.

```powershell
# 0. Bring the podman machine and wally-dev container up.
#    If wally-dev doesn't exist yet, uncomment the `podman run` line.
podman machine start
podman start wally-dev 2>$null
# if ($LASTEXITCODE -ne 0) {
#     podman run -d --name wally-dev --hostname wally-dev --network pasta `
#       -v D:\Projects\Personal\artificial-intelligence\wally:/workspace:rbind `
#       -p 8081:8081 `
#       localhost/wally-dev:latest sleep infinity
# }

# 1. Write a start script and copy it into the container, then run detached.
#    Use the system Python 3.10 + the system-installed MineStudio
#    at /usr/local/lib/python3.10/dist-packages/ (installed by the
#    Dockerfile). --viewer none suppresses the local OpenCV window
#    (the MJPEG stream IS the viewer).
$script = @'
#!/bin/bash
export PYTHONPATH=/workspace/src
export MINESTUDIO_DIR=/tmp/MineStudio
exec python3 -m wally.agent.play \
  --relay --relay-host 0.0.0.0 --relay-port 8081 \
  --checkpoint /workspace/checkpoints/checkpoint_100000.pt \
  --goal-frame /workspace/checkpoints/goal_frame1.png \
  --planner cem --viewer none \
  --record --output-dir /workspace/ag-tests/run_wood
'@
Set-Content logs\start-play.sh -Value $script -NoNewline
podman cp logs\start-play.sh wally-dev:/tmp/start-play.sh
podman exec wally-dev chmod +x /tmp/start-play.sh
podman exec wally-dev mkdir -p /workspace/ag-tests/run_wood
podman exec -d wally-dev bash -c 'setsid nohup /tmp/start-play.sh > /tmp/wally-play.log 2>&1 < /dev/null & disown'

# 2. Wait ~15s for the relay to bind, then check it:
Start-Sleep 12
podman exec wally-dev curl -s -m 3 http://localhost:8081/healthz   # -> "ok"

# 3. Open the stream in any browser on the Windows host:
#    http://localhost:8081/stream
#    You should see a 640x360 MJPEG of the agent's POV at ~30fps.

# 4. Tail the log if anything looks off:
podman exec wally-dev tail -f /tmp/wally-play.log

# 5. Stop cleanly:
podman exec wally-dev bash -c 'pkill -TERM -f "wally.agent.play"; sleep 2; pkill -KILL -f "wally.agent.play"'
#   (orphan xvfb + java processes will be reaped as <defunct> zombies
#    by the podman parent — harmless, they don't hold resources)
```

### Analyzing the recorded trajectory

If you started the agent with `--record --output-dir ...`, the full episode
is written to `<output-dir>/episode_0.npz` when the run ends. The npz has
`frames (T,64,64,3)`, `actions (T,25)`, and `events (T,)` (one dict per
step with inventory, mine_block, pickup, etc.). The fastest way to know
"did the agent do anything useful" is the bundled analyzer:

```powershell
# Copy the trajectory out of the container
podman cp wally-dev:/workspace/ag-tests/run_wood/episode_0.npz `
    ag-tests\run_wood\episode_0.npz

# Run the analyzer (verdict at the bottom: SUCCESS / FAIL with reasons)
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" `
    tools\analyze_trajectory.py ag-tests\run_wood\episode_0.npz
```

The analyzer reports per-action stats (mean over time, max, non-zero
count), camera shake magnitude, inventory spam count, mine/pickup events,
and ends with a verdict. Two common failure modes it catches:

- **CEM inventory local minimum**: `actions[:, 12] > 0.5` for > 50 steps
  means the agent got stuck opening/closing the inventory. The stream
  shows the inventory UI flashing open/closed. Fix: train more, or
  apply the `action[12]=0` mask in `src/wally/agent/loop.py` (see
  `src/wally/agent/AGENTS.md`).
- **Camera shake, no progress**: `|delta(yaw)| > 0.1` on > 50% of
  consecutive step pairs with no inventory spam and no movement means
  the model is too under-trained to commit to a direction. Fix: train
  longer (1k → 5k → 10k is the typical progression; see the table in
  root `AGENTS.md` → "Expected results by training size").

When the text report points at a step range but you also need to *see* the
frames, run `tools/extract_anomalies.py` for an anomaly-driven contact
sheet. Where the analyzer tells you *that* inv-spam happened at t=185..201,
the contact sheet shows the inventory-UI-overlay frame at t=200 plus the 2
frames before (so you can see what triggered the loop) and the 2 frames
after (whether the agent recovered). It also produces a `frames.json`
sidecar with the actual step indices per panel, which is what you want when
you're pasting the image into an LLM conversation:

```powershell
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" `
    tools\extract_anomalies.py ag-tests\run_wood\episode_0.npz
# writes ag-tests\run_wood\anomaly_contact_sheet.png + frames.json
```

The contact sheet is the offline counterpart to the live relay: same
underlying frames, but anomaly-driven selection (8 panels of 5-frame
windows around inv-spam, camera-shake, cost-spike, attack-burst, first-event,
brightness extremes, best-match-to-goal, final-frame) instead of 5 evenly
spaced ones. See `AGENTS.md` → "Project structure" → `tools/` for the full
relationship to `tools/extract_frames.py` (the even-spaced sampler, kept
for non-debug "what did the run look like" glances).

### What you should see

- The relay at `http://localhost:8081/stream` serves a `multipart/x-mixed-replace` MJPEG stream. The browser auto-refreshes each new frame (you'll see a quick flash between frames — that's normal).
- `http://localhost:8081/healthz` returns `ok` once the relay thread is up. Before that, the connection is refused.
- The Minecraft process logs benign warnings on startup: `fliteWrapper` (narrator library), `optifine/ctm/default/empty.png` (texture), `OpenAL` (sound), `Realms` (auth). All safe to ignore — see `src/collector/AGENTS.md`.

### Quick-start: shorter episodes and trajectory recording

The default `episode_timeout` is 1000 steps, which on the current checkpoint runs ~110s end-to-end (most of which is the agent sitting in a planner local minimum — see below). For a quick visual smoke-test, lower the timeout via a tiny `--config` YAML:

```yaml
# /tmp/quick.yaml
replan_interval: 4
episode_timeout: 200
resize: [64, 64]
```

```sh
python3 -m wally.agent.play \
  --relay --relay-host 0.0.0.0 --relay-port 8081 \
  --checkpoint /workspace/checkpoints/checkpoint_100000.pt \
  --goal-frame /workspace/checkpoints/goal_frame1.png \
  --planner cem --viewer none \
  --config /tmp/quick.yaml
```

200 steps finishes in ~25s on the current setup — fast enough to loop while iterating on the checkpoint or the planner.

To also persist what the agent did (frames, actions, and the `info` event subset — inventory, mine_block, pickup, break_item, craft_item, use_item, damage_dealt, health, food_level — see `src/wally/agent/buffer.py`), add `--record --output-dir <dir>`. The full episode is written to `<output-dir>/episode_0.npz` when the run ends (timeout, done, or `q`/`Esc`):

```sh
... --record --output-dir /workspace/ag-tests/run_wood
# -> /workspace/ag-tests/run_wood/episode_0.npz
```

The npz contains `frames (T,64,64,3)`, `actions (T,25)`, and `events (T,)` — load with `numpy.load(..., allow_pickle=True)`. This is what the `ag-tests/*.md` files use to assert things like "did the agent pick up wood" without re-watching the video.

### Verifying the world model is alive (`wally-plan-smoke`)

If the agent looks broken (button-spam, frozen POV, etc.), before retraining anything, sanity-check the checkpoint itself. `wally-plan-smoke` runs the CEM planner end-to-end on two synthetic frames and reports whether the output is structured or near-zero:

```sh
# On Windows (no MineStudio needed - works on CPU torch)
uv run wally-plan-smoke
# uses checkpoints/checkpoint_100000.pt and plan_smoke/{current,goal}.png by default

uv run wally-plan-smoke --checkpoint checkpoints/checkpoint_500.pt --output /tmp/probe.pt
```

Verdicts:

- `actions are essentially zero` → model is dead, retrain needed
- `low variance - likely noise` → model is unstable, retrain
- `model produced structured actions` → model is fine, look at the planner/agent loop instead

This is the cheapest possible signal of whether a fresh training run produced a usable checkpoint.

### Viewing the relay from Windows (`tools/start-play-bind.py`)

You can of course just open `http://localhost:8081/stream` in a browser, but for a proper always-on-top OpenCV window with health-check overlay, reconnect logic, and FPS readout, use the bundled viewer:

```powershell
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\Activate.ps1"
python tools\start-play-bind.py
# or
python tools\start-play-bind.py --url http://localhost:8081/stream --fullscreen
```

The window shows a green/red status dot (`LIVE` vs `WAITING`) and a rolling FPS counter, and reconnects with exponential backoff (`0.5, 1, 2, 4, 8 s`) when the relay drops — useful while the MineStudio JVM is still booting inside the container. Press `q` / `Esc` to quit.

### Why `setsid nohup ... & disown`?

A plain `podman exec wally-dev bash -c '... &'` will be killed the moment the outer `podman exec` shell exits, because the child process group is the same as the shell. `setsid` starts a new session (no controlling terminal), `nohup` ignores SIGHUP, and `disown` removes the job from the shell's job table. Together they fully detach the agent from the `podman exec` invocation, so the process keeps running after the PowerShell command returns. This is the same pattern Docker's `dockerd` uses for containerized daemons.

### The CEM inventory-stuck local minimum

A common visible behavior: the agent repeatedly opens and closes the inventory. This is a CEM local minimum — opening inventory produces a near-constant latent state (frozen camera + dim world) that the world model scores as marginally close to the goal latent. The fix is to mask the `inventory` action column out of the planner output:

```diff
--- a/src/wally/agent/loop.py
+++ b/src/wally/agent/loop.py
@@
             action = plan_actions[action_index]
+            if action.dim() == 1 and action.shape[-1] > 12:
+                action = action.clone()
+                action[12] = 0.0
```

This is a one-line demo hack — for production, add the same penalty into the CEM cost function in `src/wally/planner/plan.py` (e.g. `cost += 1e-3 * (actions[..., 12] ** 2).sum(dim=(-2, -1))`) so the planner learns to avoid the action instead of just being silenced. The current behavior of all three planners (`cem`, `gradient`, `hierarchical`) is to converge on the same local minimum; the mask breaks the loop for all of them.

## WSL2 planner performance

Inside the `wally-dev` container, the AMD ROCm torch shipped by the `rocm/pytorch` base image is broken for RDNA2 (librocdxg SDMA hang — see `docs/gpu-setup.md#wsl2-compute-status-broken`). The container's `Dockerfile` therefore swaps in a CPU-only torch:

```
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

In-WSL2 `wally-play` planners run on CPU torch:

- **CEM** (default): unaffected — pure numpy, no torch kernels.
- **Gradient MPC**: works, 10–50× slower than TheRock GPU on Windows.
- **Hierarchical**: works, similar slowdown.

This is a watch-the-agent loop, not a fast-evaluation loop. If you need benchmark-grade planning throughput, use the Windows-native TheRock setup and run `wally-plan` there.
