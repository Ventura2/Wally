## Why

Debugging wally agent runs is currently a three-hop process: run the agent, run `analyze_trajectory.py` to get a wall of text, then *separately* open `tools/extract_frames.py` to dump 5 evenly-spaced PNGs into a folder. The even-spaced PNGs are a poor match for debugging because the agent spends 99% of its time doing visually similar things (walking, looking at the same forest), so the 5 extracted frames almost always miss the exact step where the camera started thrashing, the inventory UI flickered, or the cost spiked. The user is forced to mentally align "t=185 inv>0.5 (col 12)" from the analyzer against "frame 185 looks like... a tree" from the extractor, with no visual anchor tying them together.

We need a single tool that takes an `episode_0.npz`, finds the anomalous moments in it, and produces **one labeled contact sheet** the user can paste into an LLM conversation to ask "what's wrong here?" — without separately aligning text timestamps with hand-picked PNGs.

## What Changes

- New `tools/extract_anomalies.py` CLI: takes an `episode_0.npz` and produces a single PNG contact sheet of the most interesting moments in the run, plus a `frames.json` sidecar describing each panel.
- Each panel = a horizontal 5-frame strip (the anomaly step + 2 frames before/after as context) with a one-line text label (anomaly class, step range, key metric).
- Anomaly detection runs per-step (local windows of ±3 steps), so a 6-step inventory-spam burst is detected as one cluster, not as 6 separate samples.
- Default 8 panels: inv-spam, camera-shake, cost-spike, attack-burst, first-pickup/first-event, best-match-goal, final-frame, brightness-extreme. Panel list is overridable via `--panels` flag.
- Output written next to the input npz by default (`<npz_dir>/anomaly_contact_sheet.png` + `frames.json`), so it pairs naturally with the existing `analyze_trajectory.py` workflow.
- `tools/extract_frames.py` is **not** modified or removed. It stays for the "even-spaced playback" use case (e.g. quick visual sanity check of a successful run).

## Capabilities

### New Capabilities
- `agent-anomaly-viewer`: offline post-hoc visualization of agent trajectory anomalies. Produces a single labeled contact sheet per npz, with per-step anomaly detection (inventory spam, camera shake, cost spike, attack burst, event-triggered frames) and per-panel context windows. Mirrors `live-agent-viewer` (real-time) as its offline counterpart.

### Modified Capabilities
None. No existing spec requirements change. `live-agent-viewer` stays real-time only; `trajectory-validator` stays focused on shard validation. This is a brand-new capability, not a delta on an existing one.

## Impact

- **New code**: `tools/extract_anomalies.py` (~200-300 lines), one optional helper module under `tools/_anomaly_scorers.py` for the per-step detectors (kept separate to keep the CLI thin and the detectors individually testable).
- **Dependencies**: existing `numpy`, `PIL.Image`, `PIL.ImageDraw` (already used in `tools/extract_frames.py` and `src/wally/agent/loop.py` no — actually let me check; if `PIL.ImageDraw` is not yet used, this is the first use). No new pip dependencies.
- **No agent runtime change**: this is a pure offline analysis tool. The recorder (`src/wally/agent/buffer.py`), the loop (`src/wally/agent/loop.py`), and the play CLI are unchanged. The tool reads `episode_0.npz` after the run, exactly like `analyze_trajectory.py`.
- **Test surface**: new `tests/test_extract_anomalies.py` with synthetic npz fixtures (controlled inv-spam burst, camera-shake burst, cost spike) to verify the scorer + cluster picker.
- **No CLI entry point added to pyproject.toml**: follows the `tools/*.py` ad-hoc-script convention used by `analyze_trajectory.py`, `extract_frames.py`, `compare_camera_fix.py`. Users run `python tools/extract_anomalies.py <npz>` directly.
