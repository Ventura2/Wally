## Context

The wally agent produces a `episode_0.npz` per run via `src/wally/agent/buffer.py:TrajectoryBuffer` (`to_dict()` at `buffer.py:63-79`). The npz contains `frames: (T, 64, 64, 3) uint8`, `actions: (T, 25) float32`, optional `events: (T,) object`, and (when the planner records SCSA data) `costs`, `scsa_z_H`, `scsa_costs`, `scsa_l2_costs`, `scsa_z_g`.

Two existing tools consume this npz for offline analysis:

- `tools/analyze_trajectory.py` — text-only report: action stats, cost progression, camera-shake metrics, event detection, SCSA spearman, final verdict. ~470 lines, has rich per-trajectory statistics but **no per-step scorer** and **no image output**.
- `tools/extract_frames.py` — saves 5 evenly-spaced frames as PNGs (one upscaled 4×). Stateless, no scoring, no event detection.

These two tools are **never run together automatically** and their outputs aren't aligned. To debug a run, the user runs `analyze_trajectory.py`, reads the text, then *separately* runs `extract_frames.py` and mentally aligns "t=185 inv>0.5" against the frame dump. Worse: even-spaced sampling almost never lands on the actual anomaly step, so the user can't see the moment the camera started thrashing — they only see trees.

We need a single tool that **finds** the anomaly steps and **shows** them as one image. This is the offline counterpart to `live-agent-viewer` (which is real-time only, see `openspec/specs/live-agent-viewer/spec.md`).

## Goals / Non-Goals

**Goals:**
- One `python tools/extract_anomalies.py <npz>` invocation produces one PNG that an LLM can be shown alongside the text from `analyze_trajectory.py`.
- The PNG *labels* each panel with the anomaly class and step range so the LLM can refer to specific panels by name ("in panel 3...").
- The PNG *contextualises* each panel with ±2 frames around the anomaly so the LLM can see what triggered it and whether the agent recovered.
- No agent runtime change. Pure offline analysis. The tool is a `tools/*.py` ad-hoc script like `analyze_trajectory.py` and `extract_frames.py` — no pyproject.toml entry, no CLI command.

**Non-Goals:**
- Real-time / streaming output. The npz is already on disk by the time this tool runs. (The `live-agent-viewer` already covers real-time.)
- Replacing `analyze_trajectory.py`. The two tools are complementary: the text report gives the numbers, the contact sheet gives the visual ground truth. This change does not call analyze from within extract — they remain independent entry points.
- Replacing `extract_frames.py`. The even-spaced sampler is the right tool for "show me what the run looked like in general" (e.g. for a successful run you want to share); the contact sheet is the right tool for "show me what went wrong." Both stay.
- Video / GIF output. Out of scope. The whole point of this tool is "10-ish static frames for an LLM" — the user explicitly wants the low-fps static-image form.
- Train-side shard inspection. `trajectory-validator` already covers dataset inspection. This tool is for `episode_0.npz` agent runs only.

## Decisions

### D1: 4×2 layout, 5-frame strip per panel

Each panel is a 1×5 horizontal strip of the anomaly step + 2 frames before/after. 8 panels tile as 4 columns × 2 rows. Final PNG: 4 × 256 × 5 = 1280 wide, 2 × (256 + 32 label) = 576 tall, with a 16px black border. Total ~1280×576.

```
PANEL (one of 8)
┌────┬────┬────┬────┬────┐
│t-2 │t-1 │ t  │t+1 │t+2 │   5 × 64×64 frames, upscaled 4× to 5 × 256×256
└────┴────┴────┴────┴────┘
 INV SPAM t=55..67, cost=2.31  ← 1-line label, 14px, white-on-black
```

**Why 5 frames (not 3, not 7):** At ~9 agent steps/sec, ±2 gives ~0.44s of pre-context and ~0.44s of post-context. Enough to see what triggered the anomaly (the 2 frames before) and whether the agent recovered (the 2 frames after). ±1 (3 frames) misses the trigger; ±3 (7 frames) eats too much horizontal space and dilutes the anomaly frame.

**Why 4×2 (not 2×4, not 8×1):** 4×2 is roughly 2.2:1 aspect ratio, fits a typical chat-window aspect ratio better than 1:4 (8×1) or 1:2 (2×4). Matches the existing 1024×512 `contact_sheet.png` in `ag-tests/run_wood_1k/`.

### D2: Per-step scorer + greedy cluster picker (not even sampling)

Per-step anomaly score is computed first; then nearby high-score points are merged into panels. This is the core difference from `extract_frames.py` (which uses uniform `i * (T-1) / 9`).

```
per-step signals (each is a 1D array of length T)
  inv_signal[t]   = 1.0 if actions[t, 12] > 0.5 else 0.0
  attack_signal[t]= 1.0 if actions[t, 7] > 0.5 else 0.0
  cost_signal[t]  = costs[t] / max(costs)            if costs recorded
  brightness[t]   = frames[t].mean() / 255
  motion[t]       = |diff(t, t-1)| / 255             per-step |frame diff|
  cam_flip[t]     = 1.0 if pitch[t-1..t+1] sign-flips AND yaw[t-1..t+1] sign-flips

per-class scorers (each returns a list of cluster centers)
  inv_spam      = cluster consecutive steps where inv_signal == 1, length >= 5
  attack_burst  = cluster consecutive steps where attack_signal == 1, length >= 3
  cost_spike    = single frame at argmax(cost_signal)
  camera_shake  = cluster consecutive steps where cam_flip == 1, length >= 4
  brightness    = single frame at argmax and argmin of brightness
  first_event   = first step where events[t] has mine_block | pickup | non-empty inventory
  best_match    = argmin of MSE(frames[t], goal_resized)   if --goal-frame
  final_frame   = last frame, always
```

Cluster picker: greedy, `min_gap=20` steps. If two cluster centers are within 20 steps, keep only the higher-scored one. This stops "panel 1: inv spam t=55-60" and "panel 2: inv spam t=62-67" from being separate panels.

**Why greedy + fixed gap (not DP or hierarchical clustering):** the 8 default anomaly classes are well-separated in time in practice (a cost spike is unlikely to land in the middle of an inv-spam burst). A 20-line greedy loop is sufficient; DP / hierarchical clustering would be over-engineering. If the empirical distribution changes we can revisit.

### D3: 8 default panels, configurable via `--panels`

Default panel set is the union of all 8 anomaly classes above, capped at 8 panels. `--panels inv_spam,camera_shake,best_match` lets the user request a subset. If more than 8 panels are requested the tool warns and shows the first 8.

**Why a fixed 8 and not "as many as needed":** The contact sheet is meant to be pasted into an LLM conversation. 8 panels × 5 frames = 40 frames, which renders fast and stays under typical 1500-token image budgets. More than 8 panels and the LLM has too much to look at; the user can re-run with `--panels` to zoom in.

### D4: PIL `ImageDraw` text labels, no extra fonts

Each panel gets a 1-line label below the strip: `<CLASS_NAME> t=<start>..<end> cost=<X.XX>`. Text is drawn with `ImageDraw.text(..., fill=(255,255,255))` on a black 32px-tall strip. Default PIL font (no font file path arg) — 14px-ish. If the default font is too small, we can ship a `.ttf` later, but for v1 the default is fine and avoids a new dependency.

**Why no matplotlib / no real font:** matplotlib is overkill for 8 lines of text, and adding `matplotlib` as a dependency for one tool violates the project rule "no new pip dependencies for a debugging tool." Default PIL font is monospace-ish on most systems and renders fast.

### D5: `frames.json` sidecar alongside the PNG

```
{
  "npz": "ag-tests/run_wood_1k/episode_0.npz",
  "n_steps": 600,
  "panels": [
    {
      "panel_id": 1,
      "anomaly_class": "inv_spam",
      "label": "INV SPAM t=55..67 (12 steps) cost=2.31",
      "window": [53, 54, 55, 56, 57],
      "score": 1.0
    },
    ...
  ]
}
```

**Why JSON (not markdown, not CSV):** LLM-friendly (one-line per panel, easy to grep), machine-parseable for follow-up tools (e.g. a future "LLM summarizes the contact sheet" step), and matches the `np.load` style already used by the rest of the codebase. The `window` array is the actual step indices drawn in the strip, so a follow-up tool can re-extract the same panels at higher resolution.

### D6: Output written next to the input npz by default

Default output paths:
- `<npz_dir>/anomaly_contact_sheet.png`
- `<npz_dir>/frames.json`

Override with `--out-png PATH --out-json PATH`. Default mirrors `tools/analyze_trajectory.py`'s convention of reading `<npz_dir>/episode_0.npz` and writing diagnostics into the same dir.

### D7: Scorers split into `tools/_anomaly_scorers.py`, CLI in `tools/extract_anomalies.py`

The scorers (`_inv_spam`, `_camera_shake`, `_cost_spike`, etc.) are pure functions taking numpy arrays and returning a list of `(step, window, label, score)`. They live in a separate module so the test file (`tests/test_extract_anomalies.py`) can import them directly without going through argparse + PIL. The CLI module just orchestrates: load npz → call scorers → cluster pick → tile + label → save.

**Why split (not one file):** `analyze_trajectory.py` is 474 lines and mixes "load npz" with "compute per-trajectory metrics" with "format text report." A new `extract_anomalies.py` of similar size would be hard to test. Splitting scorers into a separate module follows the existing project pattern (`tools/compare_camera_fix.py` is small, but `tools/latent_cluster_analysis.py` is multi-module).

## Risks / Trade-offs

- **Default PIL font may render text at inconsistent size on different OSes** → Mitigation: ship the label text as a 14-pixel-tall black strip with a fixed pixel offset; document that the font is the system default. If a real TTF is needed later, we can add `tools/_fonts/` and a font path arg.

- **8 panels may not be enough for episodes with many distinct anomaly types** (e.g. a run that has 3 inv-spam bursts + 2 cost spikes) → Mitigation: greedy cluster picker merges nearby points into one panel; the user can re-run with `--panels inv_spam,cost_spike,attack_burst,...` to swap panels. Cap of 8 is intentional — this is a glance tool, not a full report.

- **Cluster picker with `min_gap=20` may merge distinct anomalies that happen to be close** → Mitigation: 20 steps is ~2.2s of game time; in practice inv-spam and cost-spike are well-separated by class. If a real case shows over-merging, we can make `min_gap` a `--min-gap N` flag.

- **Goal-MSE panel is not useful if no `--goal-frame` is passed** → Mitigation: the panel is silently dropped (not shown as an empty cell) when no goal is provided. The remaining 7 (or 8 minus best_match) panels still render correctly. The `frames.json` records whether the panel was included.

- **5-frame strip at 4× upscale = 1280×512. Larger than 4×2 grid of 64×64 strips would be 256×128** → Mitigation: 1280×576 is still well under typical 4096×4096 PIL limits and renders in <1s. If a user wants smaller PNGs, add `--scale 2` flag later.

- **Two anomaly classes may produce overlapping panels (e.g. camera-shake and brightness-extreme hit the same step)** → Mitigation: the cluster picker uses `anomaly_class` to dedupe — if the same step wins in two classes, the higher-scored class wins. The other panel is replaced by the next best unused anomaly. Logged to stdout so the user knows the swap happened.

## Migration Plan

None. This is a brand-new tool. No existing behavior changes. The 5-frame `extract_frames.py` and the text-only `analyze_trajectory.py` are unchanged. The tool reads the same `episode_0.npz` format they do.

## Open Questions

- **Should the contact sheet be embedded in `analyze_trajectory.py`'s verdict section?** The user can already run the two tools sequentially; auto-embedding would be a v2 change. For v1 keep them independent entry points. Revisit after we see the v1 in use.
- **Should `--scale 1` (native 64×64) be supported for users on slow networks?** Not in v1. 4× is the convention. Add the flag if requested.
- **What if `costs` is not in the npz (older runs without SCSA recording)?** The `cost_spike` panel is silently dropped, no error. Same for `events`: if events are missing, the `first_event` panel is dropped.
