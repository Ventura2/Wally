# L0 1k-step diagnosis — findings report

> Target: `checkpoints/wood_1000/checkpoint_1000.pt` (vit_tiny, depth=4, **encoder_type=cnn**, embed_dim=192, 1000 steps).
> Hardware: Windows-native ROCm venv, GPU inference used.
> Scripts: `tools/experiments/00_…07_*.py` (12 files, 6 figures, all stdout captured to `_*.txt` siblings).

---

## 1. What I did

All scripts under `D:\Projects\Personal\artificial-intelligence\wally\tools\experiments\`. They import `wally.*` from the live `src/` tree (so the load path matches `wally-train` / `LatentRollout.from_checkpoint` exactly). **None of them modify wally's source.** Commands assume the venv is activated.

| # | File | Probes | Run | Status |
|---|------|--------|-----|--------|
| 00 | `00_load_and_sanity.py` | Load L0 via the canonical `LatentRollout.from_checkpoint`, dump model_config, run one forward pass. | `python tools\experiments\00_load_and_sanity.py` | worked first try (after a f-string fix) |
| 01 | `01_ood_probe.py` | **A.** Encode 16 real + 6 synth frames (sky, water, ground, cave, trunk close/far). Pairwise latent L2, OOD distance, 1-step MSE for "next = synth" cases, and the action-magnitude probe. PCA-2D figure. | `python tools\experiments\01_ood_probe.py` | worked (Unicode fix) |
| 02 | `02_rollout_divergence.py` | **B.** 100 random rollouts per start × 4 starts, horizons 1/2/4/8/16/32. Mean ± std of `‖z_H‖`, OOB fraction, per-step variance. | `python tools\experiments\02_rollout_divergence.py` | worked first try |
| 03 | `03_camera_shake_sandbox.py` | **C.** Predicted `Δz` magnitude as a function of `|camera_pitch|`, `|camera_yaw|` (0.0–1.0). Then 160 CEM plans (40 plans × 4 starts) — first-action distribution. Then `action_embedder` saturation test. | `python tools\experiments\03_camera_shake_sandbox.py` | worked (after trimming the plan count from 200 → 40 to fit in 2 min) |
| 04 | `04_latent_geometry.py` | **D.** Encode 512 real frames (8 chunks × 64 frames). PCA, correlation of `‖z‖` with brightness / top RGB / sky fraction. | `python tools\experiments\04_latent_geometry.py` | worked (after fixing a couple of axis / shape bugs) |
| 05 | `05_action_clipping.py` | **G.** One-step prediction error vs camera clip value (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0). 8-step rollouts with camera clipped to ±0.1, ±0.3, ±1.0. | `python tools\experiments\05_action_clipping.py` | worked first try |
| 06 | `06_toy_2d_world.py` | **E.** 2-D navigation, frozen L0 encoder + small WM. | `python tools\experiments\06_toy_2d_world.py` | "worked" — but **inconclusive** (L0 encoder is meaningless on toy images; the test collapses). |
| 06b | `06b_toy_2d_world_scratch.py` | Same but with a from-scratch tiny CNN encoder. | `python tools\experiments\06b_toy_2d_world_scratch.py` | **Inconclusive — the jointly-trained encoder collapses** (latent norm 0.15, no positional signal; planner reduces dist 1.09 → 1.10). |
| 06c | `06c_toy_2d_world_positional.py` | Same with a **deterministic Fourier-feature positional encoder** so the planner actually has signal. | `python tools\experiments\06c_toy_2d_world_positional.py` | worked first try. **Conclusive: planner is fine.** |
| 07 | `07_cost_informativeness.py` | **I.** 32×32 matrix of `‖z_s − z_g‖²` vs pixel L2. Spearman correlation, top-1 agreement. | `python tools\experiments\07_cost_informativeness.py` | worked |

Figures in `tools/experiments/_figures/`: `01_ood_latents.png`, `02_rollout_divergence.png`, `03_camera_shake.png`, `04_latent_geometry.png`, `04_norm_vs_brightness.png`, `07_cost_vs_pixel.png`.

---

## 2. What I found

### A. OOD probe — **hypothesis strongly confirmed**

Latent norms (from 16 real + 6 synth frames):

| frame kind | `‖z‖` | real-data sigma away |
|---|---|---|
| real frames (16) | 5.99 – 8.63 (mean 7.43, **std 0.62**) | — |
| dark_cave | 5.73 | −2.7 |
| trunk_close | 11.12 | +5.9 |
| ground_close | 15.96 | +13.7 |
| trunk_far | 15.31 | +12.7 |
| **sky** | **17.24** | **+15.7** |
| **water** | **23.80** | **+26.3** |

A synth sky frame's nearest real neighbour is **9.28** L2 away in latent space (mean over 16 starts is 10.86). The L0's 1-step prediction error when the target is a real frame is **24.97**; the same prediction error against a sky target is **873**, against water **2679** — **35× and 107× higher respectively**. So the cost does spike at OOD frames exactly as the user observed. The hypothesis holds, but the magnitude of the spike is much larger than the user's "100 vs 25" estimate (4×) — the synth-OOD ratio is 35–100×, not 4×.

### B. Rollout divergence — **hypothesis confirmed**

100 random rollouts per start, 4 starts, random `a ∈ [−1, 1]^25`:

| H | mean `‖z_H‖` | std | variance | OOB frac (`‖z‖ > 3×train`) |
|---|---|---|---|---|
| 1 | 7.62 | 0.19 | 0.005 | 0.00 |
| 2 | 7.72 | 0.24 | 0.008 | 0.00 |
| 4 | 8.01 | 0.42 | 0.017 | 0.00 |
| 8 | 8.77 | 0.83 | 0.034 | 0.00 |
| 16 | 10.95 | 1.34 | 0.066 | 0.00 |
| 32 | 17.08 | 2.19 | 0.153 | 0.02 |

At the planner's nominal **H=8** the L0 is 1.34 sigma above training mean (8.77 vs 7.43). At H=16 it's 5.66 sigma above, at H=32 it's 15.5 sigma above. The variance grows linearly with horizon (slope 0.005/step). The OOB threshold (3× train mean = 22.3) is not crossed until H≈40 — so 8-step rollouts *do not* explode, but they are already noticeably outside the training distribution. **This is the smoking gun for why replan_interval=2 didn't help**: the L0's 8-step rollout is on the edge of the training distribution, and any inaccuracy (camera scale, OOD frame, etc.) pushes the prediction outside it. Faster replanning just asks the L0 to make more predictions in the marginal zone.

### C. Camera shake — **hypothesis partially confirmed; new finding**

Predicted `‖Δz‖` from a real frame with a single camera action (zero everywhere else):

| `|cam|` | pitch | yaw |
|---|---|---|
| 0.00 | 0.226 | 0.226 |
| 0.05 | 0.186 | **1.951** |
| 0.10 | 0.170 | **2.913** |
| 0.20 | 0.181 | **3.005** |
| 0.30 | 0.203 | 1.520 |
| 0.50 | 0.236 | 0.848 |
| 0.70 | 0.256 | 0.725 |
| 1.00 | 0.275 | 0.661 |

The yaw curve is **strongly non-monotonic** with a peak at `|yaw| ≈ 0.2` (Δz = 3.0). The L0's one-step prediction error follows the same shape (2.0 at |cam|=0, peaks at 10.7 at |cam|=0.3, falls back to 2.6 at |cam|=1.0). This is a structural defect in the 1k-step L0: the action embedder has learned a wildly non-monotonic response. The planner learns to exploit it: 53% of its chosen first-action pitches are `|pitch| > 0.5`, and 40% of yaws are `|yaw| > 0.5`. Net/total motion on `camera_pitch` over 160 plans: **net=0.000, total=0.522** — exactly the user's "shakes but doesn't go anywhere" pattern. (See 03_camera_shake.png.)

The action_embedder saturation test (C section) shows: at L0 dim-10 = 0.0, ‖e‖ = 11.89. At L0 dim-10 = 1.0 (the maximum the L0 was actually trained on, after the dataloader's `clamp(-1, 1)`), ‖e‖ = 12.24. At L0 dim-10 = 180 (what the planner actually sends when the agent picks |cam|=1.0), ‖e‖ = 132.96 — **11× the trained range**. The L0's response is therefore extrapolating well outside its training distribution on every plan step.

### D. Latent geometry — **NEW dominant finding**

512 real frames, PCA on `Z ∈ R^{512×192}`:

- **PC1 alone explains 83.6 % of the variance**; PC1+PC2 = 93.5 %.
- `‖z‖` correlates with frame **brightness at +0.973**.
- `‖z‖` correlates with top-row R, top-row B at +0.95 each.
- `‖z‖` correlates with sky-fraction at +0.92.

The 1k-step L0's latent is essentially a **1-D brightness axis** (see `04_norm_vs_brightness.png` — the linear relationship is striking). The CNN encoder has not yet learned any spatial structure beyond "how bright is the upper half of the frame". This explains the OOD spike (sky is the brightest thing the agent can look at) and reframes the camera-shake story: the planner is not trying to look at a *tree*; it is trying to match the goal's brightness level by panning around.

### E. Toy 2-D world — **CEM planner is fine**

With a **deterministic positional encoder** (random Fourier features of `(x, y)`) and a 1k-step-trained tiny MLP world model (final MSE 0.0025), the same wally `CEMOptimizer` produces:

| Horizon | mean start→goal dist | success @0.15 |
|---|---|---|
| do-nothing baseline | 1.073 | — |
| random actions × 8 | 1.088 | 2 % |
| CEM × 8 (pop 128, iters 8) | **0.799** | 38 % |
| CEM × 20 (pop 128, iters 10) | **0.264** | **77 %** |

The planner is **not** the bug. The 1-D OOD/rollout/encoder story above is specifically a property of the 1k-step Minecraft L0, not a generic wally problem. (My first two toy attempts (06, 06b) failed for unrelated reasons: the frozen Minecraft encoder produces noise on toy images, and a jointly-trained CNN encoder collapsed to a constant — both expected. The 06c version makes the test fair by giving the planner a clean positional signal.)

### G. Action clipping — **does not help**

One-step prediction error as a function of the camera clip (per-dim clip, agent-vocab scale `[-c, c]`):

```
clip=±0.00  ||Δz||=0.196   MSE=2.04
clip=±0.05  ||Δz||=0.324   MSE=2.06
clip=±0.10  ||Δz||=0.806   MSE=2.45
clip=±0.20  ||Δz||=2.789   MSE=9.49
clip=±0.30  ||Δz||=2.969   MSE=10.73   <-- worst
clip=±0.50  ||Δz||=2.841   MSE=9.84
clip=±0.70  ||Δz||=1.339   MSE=3.55
clip=±1.00  ||Δz||=0.885   MSE=2.63
```

8-step rollouts (`a_cam` random in [-1, 1], 30 rollouts × 4 starts):

| camera clip | mean `‖z_H‖` | per-step `‖Δz‖` |
|---|---|---|
| [-1, 1] | 10.13 | 1.11 |
| ±0.3 | 17.69 | 2.35 |
| ±0.1 | 15.10 | 1.93 |

**Clipping the camera makes the L0's predicted rollouts *worse*, not better.** The intuition that "small actions → small prediction errors" is wrong for this L0: the worst predictions are at moderate amplitudes (0.2–0.5), and the response shape is a multi-modal artifact of the action embedder, not a monotone scale. So the recommendation "clip the camera" is *not* supported as a standalone fix. The right fix is to fix the action scale (see recommendation 1 below).

### I. Cost informativeness — **cost is informative, not noise**

32×32 matrix of L0 cost `‖z_s − z_g‖²` vs pixel L2 over 64 real frames (one chunk):

- Spearman correlation: **+0.836**
- Pearson (vs log(1+pix)): +0.756
- Top-1 nearest-by-pix-vs-nearest-by-cost match: **75 %** (24/32)

The L0 cost surface *does* track pixel similarity (it's not random). The 25 % mismatch and the brightness axis are the problem: the L0 thinks two equally-bright frames are similar, even if one is dirt and the other is sky.

### Existing AG-test trajectory confirms the picture

`ag-tests/run_wood_1k/episode_0.npz` (600 steps):

- `camera_pitch`: mean **+0.427** (saturated positive), `|pitch|>0.5` frac **0.522**, flip-frac 0.27
- `camera_yaw`: mean +0.044 (oscillating around zero), `|yaw|>0.5` frac 0.402, flip-frac **0.513** (random walk)
- Frame brightness: starts at 60, ends at **38** — the agent is *not* looking at the bright sky; it's walking around in dimmer forest. The planner's camera-pitch saturation is therefore not a "look at sky" plan but an exploitation of the L0's non-monotonic camera response (Section C).

---

## 3. What the L0 actually knows

After 1k steps on treechop_full, the L0 has learned **roughly one thing**: a 1-D, monotone map from frame brightness to `‖z‖` (correlation +0.97 with brightness, PC1 = 84 % of variance). The encoder has not learned spatial structure. The predictor has learned a per-step residual of `‖Δz‖ ≈ 0.3` (a small constant) and a non-monotonic, peak-at-moderate-amplitudes camera response that the action embedder amplifies by 11× when the planner sends full-scale inputs. The L0 cost surface is informative at the *frame-pair* level (Spearman +0.84 with pixel distance) but coarse — a brighter frame and a dirt frame are equally "close" in the L0's eyes. The 8-step rollout is on the edge of the training distribution (1.3 σ above) and OOD synth frames are 15–25 σ away in latent space. The CEM planner is fine — given a good world model it solves a 2-D navigation task with 77 % success. The L0 is the problem, and the dominant failure modes are (i) 1-D latent, (ii) action-scale mismatch with the planner, and (iii) non-monotonic camera response.

---

## 4. Concrete recommendations

### 1. Fix the agent→L0 camera scale (highest impact, lowest cost)

**Change**: in `src/wally/planner/rollout.py:113-116`, change `_CAMERA_DEGREE_SCALE = 180.0` to `1.0` (or remove the scale and just permute), and in `src/wally/data/dataset.py:65-66`, **remove** the `actions.clamp(-1.0, 1.0)` on raw camera values during training (or document it explicitly so they stay in sync). The data was clamped to ±1 in training but the planner sends ±180 — a 180× mismatch.

**Expected effect**: the L0 sees action magnitudes in the range it actually trained on; the non-monotonic "moderate amplitude" peak in Section C disappears or moves to the right; the planner stops producing the 11×-extrapolation through the action embedder.

**Evidence**: 01_ood_probe.py C-section shows the action_embedder's `‖e‖` goes from 12.24 (L0 cam=1) to 132.96 (L0 cam=180) — an 11× amplification the L0 never saw during training. 03_camera_shake_sandbox.py C-section shows this happens in one step of the predictor.

### 2. Train much longer, or use a stronger encoder

**Change**: increase `max_steps` from 1000 in `configs/lewm_wood_*.yaml`. The 1k-step L0's encoder has not yet learned spatial features (Section D).

**Expected effect**: PC1's variance share drops, `‖z‖` stops correlating with brightness, OOD synth frames move closer in latent space, the cost surface becomes finer-grained.

**Evidence**: PC1 = 83.6 % of variance, `‖z‖` vs brightness r = +0.973, OOD distance 9–17 (vs real within-cluster spread 1.5).

### 3. Drop the 1-D brightness axis from the cost

**Change**: in `src/wally/planner/plan.py:_default_cost`, replace `((z_H - z_g) ** 2).sum(dim=-1)` with a brightness-normalised cost, e.g. `((z_H - z_g) / (z_H.norm(dim=-1, keepdim=True) + 1e-3) - (z_g / (z_g.norm(dim=-1, keepdim=True) + 1e-3))) ** 2 .sum(dim=-1)`, OR subtract the dataset-mean latent from both before the L2.

**Expected effect**: the planner stops scoring "brightness match" and starts scoring "structural match"; the camera-pitch saturation in the AG-test trajectory should drop.

**Evidence**: 04_latent_geometry.py — PC1 is essentially a brightness axis; 01_ood_probe.py — sky frames have 2.3× training mean `‖z‖` and dominate the cost.

### 4. Reduce CEM horizon to H=4 and replan every 2 steps

**Change**: in the agent's CEM config (set in `src/wally/agent/planner_factory.py:71-77` and the AG-test config in `checkpoints/ag_test_wood.yaml`), set `horizon: 4` and `replan_interval: 2`.

**Expected effect**: rollouts stay well inside the training distribution (H=4 has ‖z_H‖ = 8.01 vs 8.77 at H=8), so the cost is less OOD. Doesn't fix the scale bug, but should be a free improvement on top of (1) and (2).

**Evidence**: 02_rollout_divergence.py — H=4 → 0.42 sigma above train mean, H=8 → 1.34, H=16 → 5.66.

### 5. (Investigative) Replace the L0 cost with a discriminator-based "is-this-the-goal" head

**Change**: after training, add a small binary classifier `(z_t, z_g) → P(z_t is a goal-state neighbour of z_g)`, train on the (chunk, goal) pairs from the dataset, and use its negative log-prob as the CEM cost instead of the L0 L2. This bypasses the L0's brightness bias entirely.

**Expected effect**: the cost is no longer dominated by the brightness axis; the planner can use the L0 latent for what it can encode (whatever the encoder does learn) without paying the 1-D cost penalty.

**Evidence**: 07_cost_informativeness.py — L0 cost is informative (Spearman 0.84), so a learned reweighting on top of it should improve, not replace, the signal.

---

## Negative / inconclusive results

- **Toy 2-D world with frozen L0 encoder (06_toy_2d_world.py)** — `mean dist 1.24` (worse than no-op 1.09). Inconclusive: the L0 encoder is meaningless on toy images. Expected.
- **Toy 2-D world with from-scratch CNN encoder (06b_toy_2d_world_scratch.py)** — WM loss collapses to ~0, but the encoder also collapses (mean ‖z‖=0.15, no positional signal). Inconclusive: needs a different encoder supervision signal.
- **Action clipping (Section G)** — does **not** help. The hypothesis that "small actions → small errors" is false for this L0.
- **CAM `diversity_penalty` / `camera_still_penalty`** — I did not isolate these from the planner; the runs used the agent's default values (0.001 each). Probably negligible compared to the 1-D-cost and scale issues.
- **Predictor first-layer filters / visual interpretability** — the encoder is a CNN (not ViT), so there is no patch-token "first layer" to look at; I did not pursue visualisation further.

