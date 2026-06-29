# Agent Anomaly Viewer

## Purpose

Provides an offline, post-hoc visualization of agent trajectory anomalies. Given a `episode_0.npz` produced by `wally-play` (or any compatible recorded trajectory), the viewer detects per-step anomalies (inventory spam, camera shake, cost spikes, attack bursts, event-triggered moments, brightness extremes, goal-match) and renders them as a single labeled contact sheet (PNG) plus a machine-readable index (JSON) describing each panel. Mirrors `live-agent-viewer` (real-time) as its offline counterpart.

## Requirements

### Requirement: CLI entry point
The system SHALL provide a CLI script `tools/extract_anomalies.py` that accepts an `episode_0.npz` path as a positional argument and produces an anomaly contact sheet.

#### Scenario: Default invocation
- **WHEN** the user runs `python tools/extract_anomalies.py ag-tests/run_wood_1k/episode_0.npz`
- **THEN** the tool reads the npz, detects anomalies, and writes two files next to the input: `anomaly_contact_sheet.png` and `frames.json`

#### Scenario: Missing input file
- **WHEN** the user runs the tool with a path that does not exist
- **THEN** the tool exits with code 2 and prints `ERROR: npz not found: <path>` to stderr

#### Scenario: Input lacks required keys
- **WHEN** the input npz is missing the `frames` or `actions` keys
- **THEN** the tool exits with code 2 and prints `ERROR: npz missing required key '<key>'` to stderr

### Requirement: Per-step anomaly detection
The system SHALL compute per-step anomaly scores for at least 8 anomaly classes and return a list of cluster centers (each: anomaly class, central step, ±2-step window, label string, score).

#### Scenario: Inventory-spam cluster detected
- **WHEN** the input actions contain a contiguous run of `actions[:, 12] > 0.5` for at least 5 steps
- **THEN** the inv-spam scorer returns one cluster centered at the middle of that run with a window covering `±2` steps around the center and a label of the form `INV SPAM t=<start>..<end> (<N> steps)`

#### Scenario: Inventory-spam ignored if too short
- **WHEN** `actions[:, 12] > 0.5` is true for fewer than 5 contiguous steps
- **THEN** the inv-spam scorer returns an empty list (no panel)

#### Scenario: Camera-shake burst detected
- **WHEN** the pitch and yaw action columns both sign-flip within a rolling 3-step window for at least 4 contiguous steps
- **THEN** the camera-shake scorer returns one cluster covering the burst

#### Scenario: Cost spike detected
- **WHEN** the npz contains a `costs` array
- **THEN** the cost-spike scorer returns a single cluster at `int(costs.argmax())`

#### Scenario: Cost spike skipped if not recorded
- **WHEN** the npz does not contain a `costs` array
- **THEN** the cost-spike scorer returns an empty list (no error)

#### Scenario: Attack burst detected
- **WHEN** the input actions contain a contiguous run of `actions[:, 7] > 0.5` for at least 3 steps
- **THEN** the attack-burst scorer returns one cluster

#### Scenario: First event detected
- **WHEN** the npz contains an `events` array and at least one event has a non-empty `mine_block`, `pickup`, or inventory with a non-none item
- **THEN** the first-event scorer returns a single cluster at the first such step

#### Scenario: Brightness extremes detected
- **WHEN** the per-step mean brightness (frame mean over HWC) is computed for all steps
- **THEN** the brightness scorer returns two clusters: one at `argmax` and one at `argmin`, with windows `±2`

#### Scenario: Best-match-to-goal detected when goal provided
- **WHEN** the user passes `--goal-frame PATH` and the path exists
- **THEN** the best-match scorer computes MSE between each frame and the resized goal image and returns a single cluster at the argmin step

#### Scenario: Best-match-to-goal skipped when no goal
- **WHEN** the user does not pass `--goal-frame`
- **THEN** the best-match scorer returns an empty list (no error, no panel)

#### Scenario: Final frame always included
- **WHEN** the npz has at least 1 frame
- **THEN** the final-frame scorer returns a single cluster at `T - 1`

### Requirement: Cluster deduplication
The system SHALL merge clusters whose central step is within `min_gap=20` steps of a higher-scored cluster, keeping the higher-scored cluster and dropping the lower.

#### Scenario: Two inv-spam bursts close together
- **WHEN** the scorer returns two inv-spam clusters at steps 55 and 67 (12 steps apart, less than `min_gap=20`)
- **THEN** the picker keeps the higher-scored one and drops the other; the final panel list contains one inv-spam panel, not two

#### Scenario: Two distinct anomaly classes far apart
- **WHEN** the inv-spam cluster is at step 55 and the cost-spike cluster is at step 400
- **THEN** both clusters survive (gap is much greater than `min_gap=20`)

### Requirement: 4×2 contact sheet layout
The system SHALL render up to 8 panels in a 4-column × 2-row grid. Each panel SHALL be a horizontal strip of 5 frames (anomaly step + 2 before + 2 after) at the native 64×64 frame size. The full grid is 1280 pixels wide × 192 pixels tall (5 × 64 = 320 per strip × 4 columns; 64 frame + 32 label = 96 per row × 2 rows). The full image including the 8-pixel black border on all sides SHALL be 1296 × 208 pixels.

#### Scenario: Default layout for 8 panels
- **WHEN** 8 anomaly clusters are selected
- **THEN** the output PNG is 1296×208 with 8 panels in a 4×2 grid, each panel showing 5 frames at 64×64 and a 1-line label below

#### Scenario: Layout dimensions are deterministic
- **WHEN** the tool is called with any set of panels (1 to 8)
- **THEN** the rendered image is always 1296×208 pixels (black cells fill empty slots)

#### Scenario: Fewer than 8 panels
- **WHEN** fewer than 8 clusters are detected (e.g. an episode with no cost spike and no events)
- **THEN** the remaining grid cells are filled with black; the JSON records the actual panel count

#### Scenario: More than 8 clusters detected
- **WHEN** more than 8 clusters are detected after dedup
- **THEN** the tool prints a warning to stderr and renders only the first 8 (in scorer-declaration order); the JSON records the dropped clusters under a `truncated` key

### Requirement: Per-panel text label
The system SHALL draw a 1-line label below each frame strip. The label SHALL contain the anomaly class name, the step range of the underlying cluster, and (when available) the cluster's summary metric.

#### Scenario: Inv-spam label format
- **WHEN** the panel is an inv-spam cluster spanning steps 55..67
- **THEN** the label text reads `INV SPAM t=55..67 (12 steps)` drawn in white on a black 32-pixel-tall strip below the 5 frames

#### Scenario: Cost-spike label includes cost value
- **WHEN** the panel is a cost-spike cluster at step 400 with `costs[400] = 2.31`
- **THEN** the label text reads `COST SPIKE t=400 cost=2.31`

### Requirement: JSON sidecar
The system SHALL write a `frames.json` file next to the contact sheet PNG. The JSON SHALL be a top-level object with `npz`, `n_steps`, and `panels` keys. Each panel entry SHALL have `panel_id`, `anomaly_class`, `label`, `window` (list of 5 step indices), and `score`.

#### Scenario: JSON validates against expected schema
- **WHEN** the tool completes successfully
- **THEN** the JSON parses with `json.load` and contains exactly the expected keys per panel

#### Scenario: Window reflects the actual rendered frames
- **WHEN** the panel is a cost spike at step 400 in a 600-step episode
- **THEN** the `window` field is `[398, 399, 400, 401, 402]` (5 steps, centered on 400, clamped to `[0, T-1]`)

#### Scenario: Goal match omitted from JSON when not requested
- **WHEN** the user does not pass `--goal-frame`
- **THEN** no panel entry with `anomaly_class == "best_match"` appears in the JSON

### Requirement: Configurable panel selection
The system SHALL accept a `--panels CSV` argument listing the anomaly classes the user wants to see.

#### Scenario: Subset of defaults
- **WHEN** the user runs `python tools/extract_anomalies.py <npz> --panels inv_spam,camera_shake,best_match --goal-frame g.png`
- **THEN** only those 3 panels are rendered (in the listed order); the remaining 5 cells are black; the JSON contains exactly 3 panel entries

#### Scenario: Unknown panel class
- **WHEN** the user passes `--panels inv_spam,foo`
- **THEN** the tool exits with code 2 and prints `ERROR: unknown panel class 'foo'` to stderr

#### Scenario: Empty panel list
- **WHEN** the user passes `--panels` with an empty value
- **THEN** the tool exits with code 2 and prints `ERROR: --panels must list at least one anomaly class` to stderr

### Requirement: Output path overrides
The system SHALL default to writing the PNG and JSON next to the input npz. The user MAY override the output paths with `--out-png PATH` and `--out-json PATH`.

#### Scenario: Default output paths
- **WHEN** the user runs `python tools/extract_anomalies.py ag-tests/run_wood_1k/episode_0.npz`
- **THEN** the tool writes `ag-tests/run_wood_1k/anomaly_contact_sheet.png` and `ag-tests/run_wood_1k/frames.json`

#### Scenario: Custom output paths
- **WHEN** the user passes `--out-png /tmp/sheet.png --out-json /tmp/index.json`
- **THEN** the tool writes to those exact paths and does not write next to the input

### Requirement: Pure offline operation
The system SHALL NOT modify the input npz, SHALL NOT call into the agent runtime (`src/wally/agent/`), and SHALL NOT write any files other than the contact sheet PNG and the JSON sidecar (unless `--out-png` / `--out-json` override is given).

#### Scenario: No side effects on input
- **WHEN** the tool completes successfully
- **THEN** the input npz file's mtime, size, and contents are unchanged

#### Scenario: No agent runtime import
- **WHEN** `tools/extract_anomalies.py` is imported
- **THEN** no module under `src/wally/agent/` is imported (the tool depends only on `numpy` and `PIL`)
