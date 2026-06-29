## 1. Scorers module

- [x] 1.1 Create `tools/_anomaly_scorers.py` with an `AnomalyCluster` dataclass (fields: `anomaly_class: str`, `center: int`, `window: list[int]`, `label: str`, `score: float`)
- [x] 1.2 Implement `score_inv_spam(actions) -> list[AnomalyCluster]` that returns one cluster per contiguous run of `actions[:, 12] > 0.5` with length >= 5; cluster center = midpoint of the run; window = `[center-2, center+1, center, center+1, center+2]` clamped to `[0, T-1]`; label = `INV SPAM t=<start>..<end> (<N> steps)`
- [x] 1.3 Implement `score_camera_shake(actions) -> list[AnomalyCluster]` that detects bursts where both pitch and yaw sign-flip within a rolling 3-step window for >= 4 contiguous steps; label = `CAMERA SHAKE t=<start>..<end>`
- [x] 1.4 Implement `score_cost_spike(actions, costs) -> list[AnomalyCluster]` that returns a single cluster at `int(costs.argmax())`; label = `COST SPIKE t=<step> cost=<value:.2f>`; returns `[]` if `costs is None`
- [x] 1.5 Implement `score_attack_burst(actions) -> list[AnomalyCluster]` that returns one cluster per contiguous run of `actions[:, 7] > 0.5` with length >= 3; label = `ATTACK BURST t=<start>..<end> (<N> steps)`
- [x] 1.6 Implement `score_first_event(events) -> list[AnomalyCluster]` that returns a single cluster at the first step with non-empty `mine_block` / `pickup` / non-none inventory item; label = `FIRST EVENT t=<step>`; returns `[]` if `events is None` or no qualifying event
- [x] 1.7 Implement `score_brightness(frames) -> list[AnomalyCluster]` that returns two clusters: one at `brightness.argmax()` and one at `brightness.argmin()`; labels = `BRIGHTNESS MAX t=<step> val=<X.XX>` and `BRIGHTNESS MIN t=<step> val=<X.XX>`; both windows clamped
- [x] 1.8 Implement `score_best_match(frames, goal_img) -> list[AnomalyCluster]` that returns a single cluster at `argmin(MSE(frames[t], goal))`; label = `BEST MATCH GOAL t=<step> mse=<X.XX>`; returns `[]` if `goal_img is None`
- [x] 1.9 Implement `score_final_frame(frames) -> list[AnomalyCluster]` that always returns one cluster at `T - 1`; label = `FINAL FRAME t=<step>`
- [x] 1.10 Implement `dedup_clusters(clusters, min_gap=20) -> list[AnomalyCluster]` that greedily removes clusters whose center is within `min_gap` of an already-kept cluster with a higher score; preserves declaration order of surviving classes

## 2. Renderer

- [x] 2.1 Add `render_contact_sheet(frames, panels, scale=4) -> PIL.Image` in `tools/_anomaly_scorers.py` (or split into `tools/_anomaly_render.py` if file grows past 200 lines); 4×2 grid, each cell = 5 frames at 256×256 (when `scale=4`) with a 32-pixel-tall black label strip below; cells with no panel are filled solid black
- [x] 2.2 Add `draw_label(image, x, y, width, text) -> None` using `PIL.ImageDraw.text` with default font, fill=(255,255,255), on the 32-pixel-tall black strip; text is left-padded by 8 pixels
- [x] 2.3 Add `serialize_frames_json(npz_path, n_steps, panels, truncated=None) -> dict` returning the JSON-serializable structure required by the spec (keys: `npz`, `n_steps`, `panels`, optional `truncated`)

## 3. CLI (`tools/extract_anomalies.py`)

- [x] 3.1 Add argparse with positional `npz` (required), `--goal-frame PATH` (optional), `--panels CSV` (default: 8-class list), `--out-png PATH` (default: `<npz_dir>/anomaly_contact_sheet.png`), `--out-json PATH` (default: `<npz_dir>/frames.json`); exit 2 on bad input
- [x] 3.2 Load npz with `np.load(allow_pickle=True)`; exit 2 with `ERROR: npz missing required key '<key>'` if `frames` or `actions` is absent; exit 2 with `ERROR: npz not found: <path>` if the path is missing
- [x] 3.3 Call each requested scorer with the appropriate inputs; collect clusters; run `dedup_clusters`; truncate to first 8 with a stderr warning; render the contact sheet; write the PNG and JSON
- [x] 3.4 Validate `--panels` against the known class set; exit 2 with `ERROR: unknown panel class '<name>'` on unknown entries; exit 2 on empty `--panels`

## 4. Tests (`tests/test_extract_anomalies.py`)

- [x] 4.1 Add a `make_fake_npz(tmp_path, n_steps, **overrides)` fixture that writes a 64×64×3 frames array and a 25-dim action array, optionally a `costs` array and an `events` array; returns the path
- [x] 4.2 Add one test per scorer asserting the expected cluster center, window, and label for a controlled input (e.g. inv spam set on a known range, cost spike at a known index)
- [x] 4.3 Add a `test_dedup_merges_close_clusters` that injects two inv-spam bursts 12 steps apart and asserts the second is dropped
- [x] 4.4 Add a `test_render_produces_4x2_grid` that runs the renderer on 8 synthetic panels and asserts the output image is 1296×208 (1280×192 grid + 8px border on all sides)
- [x] 4.5 Add a `test_panels_flag_filters_output` that runs with `--panels inv_spam,cost_spike` and asserts the JSON contains exactly 2 panel entries
- [x] 4.6 Add a `test_missing_keys_exit_code_2` that writes a npz without `actions` and asserts the CLI exits 2 with the expected stderr text
- [x] 4.7 Add a `test_goal_frame_omitted_when_not_provided` that runs without `--goal-frame` and asserts no `best_match` panel appears in the JSON

## 5. Smoke tests on real npz

- [x] 5.1 Run `python tools/extract_anomalies.py ag-tests/run_wood_1k/episode_0.npz` from the project root; verify the output PNG and JSON exist and the JSON has the expected number of panels (this run has 0 inv-spam, 0 cost-spike, 0 events — expect inv_spam and brightness and final_frame panels at minimum)
- [x] 5.2 Run on `ag-tests/run_wood_5k_trm/episode_0.npz` (which has `costs` and SCSA data); verify the cost_spike panel is present
- [x] 5.3 Run on `ag-tests/run_wood10k_l1_5k_tree_g1/episode_0.npz` (3000 steps) to verify the tool handles long episodes without error and the min_gap dedup works as expected

## 6. Documentation

- [x] 6.1 Add a one-paragraph reference to `tools/extract_anomalies.py` under the `tools/` section of `AGENTS.md`, noting the anomaly contact sheet output (PNG + JSON) and the relationship to `tools/extract_frames.py` (even-spaced, kept for non-debug use)
- [x] 6.2 Update `src/wally/agent/AGENTS.md` "Known planner local minimum: inventory-stuck" section to point at the new tool as the recommended way to visualize the inv-spam pattern (currently the section points at the relay stream)
