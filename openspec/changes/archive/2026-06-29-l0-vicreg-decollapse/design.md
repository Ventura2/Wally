## Context

The L0 LeWorldModel is trained with `combined_loss = prediction + alpha * sigreg` (`src/wally/training/losses.py:combined_loss`). The prediction term is the standard JEPA loss (`SmoothL1` between predicted and true next-step projected embeddings). The SIGReg term enforces a closed-form whitening on the encoder output so it has unit variance per dim and zero covariance — but only as a *soft* target, applied late in training, and with a small weight (`alpha: 0.1`).

Empirically, on the `treechop_full` shards, this is not enough: the L0's 192-dim latent collapses to a 1-D "brightness meter". `tools/experiments/REPORT.md` §D documents:
- PC1 explains **84% of variance**
- `‖z‖` correlates with frame brightness at **r = +0.97**
- A toy 2-D world with a similar encoder got 77% success at H=20 — the bottleneck is the encoder's collapse, not the planner or the JEPA predictor

The collapse is rooted in the loss: `prediction` only rewards "predict the next z well", and brightness is a strong predictor of next-frame brightness. SIGReg's whitening target is too weak (`alpha=0.1`, late in training) to break the shortcut.

VICReg (Variance-Invariance-Covariance Regularization, Bardes et al. 2022) directly attacks this. It adds two loss terms applied to the **projected** encoder output `z`:

- **Variance term** — hinge loss `mean(relu(gamma - std(z, dim=0)))` pushes per-dim std toward `gamma` (default 1.0). Forces every dim to carry info.
- **Covariance term** — `mean((off_diag(cov(z)))^2)` penalizes correlation between dims. Forces dims to be decorrelated.

VICReg's paper (and the official reference impl) uses `sim_weight=25.0, std_weight=25.0, cov_weight=1.0`. These weights balance the three terms at the start; we expose them as config so they can be tuned.

## Goals / Non-Goals

**Goals:**
- Force the L0's projected encoder output to use all 192 dims meaningfully
- Break the brightness shortcut so the L1 hierarchy can plan content-distinguishing subgoals (tree vs wall vs cave)
- Keep the change scoped to the L0's training loss — no runtime, planner, or L1 changes
- Add config fields so the VICReg weight is tunable per-run (smoke vs full vs LR-search)
- Add unit tests that pin the loss formula and assert non-degenerate behavior (per-dim std, off-diag covariance)

**Non-Goals:**
- Retrain the L0 (this change *enables* the retrain, but the retrain command + checkpoint comparison happens after merge)
- Retrain the L1 / L2 (downstream, separate change)
- Change the JEPA prediction loss or the SIGReg term
- Modify the planner, agent loop, or any runtime code
- Change the data pipeline or the converter

## Decisions

### 1. Apply VICReg to the **projected** encoder output, not the raw encoder output

The existing SIGReg is applied to the projected output (`emb` in `combined_loss`), and the predictor also consumes `emb`. Applying VICReg to the same tensor means:
- VICReg, SIGReg, and the predictor all see the same distribution
- The "use all dims" property is enforced at the layer the L1 hierarchy reads from
- No need to plumb the raw encoder output through the loss

**Alternative considered:** apply VICReg to the raw encoder output (before the projector). Rejected because the projector can re-collapse the dims downstream, defeating the purpose. Applying to `emb` (post-projector) ensures the L1 sees a rich latent.

### 2. Use the standard VICReg formulation (hinge on std + off-diag covariance squared)

`src/wally/training/losses.py` will get a new `vicreg_loss(z, sim_weight, std_weight, cov_weight, gamma=1.0)`:
- `std = z.std(dim=0)` → shape `(D,)`
- `std_loss = mean(relu(gamma - std))`
- `z_centered = z - z.mean(dim=0)`
- `cov = (z_centered.T @ z_centered) / (B - 1)` → shape `(D, D)`
- `off_diag = cov - diag(diag(cov))`
- `cov_loss = (off_diag ** 2).sum() / D`
- return `sim_weight * sim_loss + std_weight * std_loss + cov_weight * cov_loss`

The `sim_loss` (prediction loss) is the existing JEPA loss; we reuse it and add the two new terms. We pass `sim_loss` into `vicreg_loss` so the function is self-contained.

**Alternative considered:** use VICReg's exact paper formulation (with `sim_weight=25` weighting the sim term, then the std and cov terms). The paper's weighting is calibrated for their experiments; in wally the JEPA prediction loss is the dominant signal, so weighting it at 25× would drown the VICReg terms. We use the paper's `std_weight=25, cov_weight=1` and let `vicreg_weight` (a single scalar) gate the whole auxiliary term — this matches how `alpha` gates SIGReg.

### 3. Add three config fields, all default-disabled

`src/wally/config/training.py` adds:
- `vicreg_weight: float = 0.0` — the gate (like `alpha` for SIGReg). Default 0.0 means VICReg is off until a config explicitly enables it
- `vicreg_std_target: float = 1.0` — the `gamma` in the hinge
- `vicreg_cov_weight: float = 1.0` — weight of the covariance term (the variance term weight is hardcoded to 1.0 in the function, matching the paper's `std_weight=25 / cov_weight=1` ratio, but scaled down to 1.0/1.0 — see open question below)

The L0 config `configs/lewm_default.yaml` sets `vicreg_weight: 1.0, vicreg_std_target: 1.0, vicreg_cov_weight: 1.0` to enable VICReg by default for new runs. The derived `configs/lewm_wood_*.yaml` files are NOT updated in this change — the user copies the default into a new config to opt in (matches the existing pattern for `early_stop` and `use_amp`).

**Alternative considered:** enable VICReg in all existing `lewm_wood_*.yaml` configs. Rejected because (a) it would silently change baseline behaviors, (b) the user already has results from those configs and shouldn't be invalidated without an explicit opt-in, (c) the "copy the default and edit" pattern is already established in the AGENTS.md quick-start.

### 4. `vicreg_weight` is a single scalar that multiplies the auxiliary term

The auxiliary term `std_loss + cov_weight * cov_loss` is computed and then multiplied by `vicreg_weight`. This matches how SIGReg's `alpha` works: one knob to turn the regularization on/off and scale it.

`vicreg_cov_weight` is a separate knob for the std:cov ratio, defaulting to 1.0 (equal weight). The paper uses 25:1 but that's calibrated for their sim loss scale. We expose the ratio so the user can tune it without touching the function signature.

### 5. VICReg loss is computed but only added when `vicreg_weight > 0`

`combined_loss` becomes:

```python
def combined_loss(predicted, target, projected, alpha, sigreg, vicreg_weight=0.0,
                 vicreg_std_target=1.0, vicreg_cov_weight=1.0):
    sim_loss = F.smooth_l1_loss(predicted, target, beta=224.0)
    sigreg_loss = sigreg(projected)
    total = sim_loss + alpha * sigreg_loss
    metrics = {"prediction_loss": sim_loss, "sigreg_loss": sigreg_loss, "total_loss": total}
    if vicreg_weight > 0:
        vicreg = vicreg_loss(projected, vicreg_std_target, vicreg_cov_weight)
        total = total + vicreg_weight * vicreg
        metrics["vicreg_loss"] = vicreg
    metrics["total_loss"] = total
    return total, metrics
```

When `vicreg_weight = 0` (default), the behavior is bit-identical to the current code. This keeps all existing smoke tests passing without modification.

### 6. L1 hierarchy is untouched in this change

The L1's encoder is `L1Encoder(l0, D1=64)` which calls `l0.encoder(frame)` then projects. The L1's `encode_sequence()` method returns the **projected** L0 output, which is what VICReg now shapes. No L1 code changes are needed — the richer L0 latent flows through automatically.

The downstream retrain of the L1 is a **separate change** (out of scope here). The proposal mentions it so the reader knows to expect a follow-up.

## Risks / Trade-offs

- **[VICReg weight too high → training instability]** → the `std_loss` and `cov_loss` can drive the latent to extremes (e.g. all dims saturate to std=2.0). **Mitigation:** the smoke test runs 100 steps and asserts the total_loss decreases monotonically; a bad weight causes loss divergence and fails the test.

- **[VICReg weight too low → no effect on collapse]** → the loss is overwhelmed by the JEPA prediction term. **Mitigation:** the proposal's acceptance criteria include running the new L0 and measuring PC1 variance; the user can iterate on the weight if PC1 is still > 70%.

- **[Default-disabled flag might be forgotten]** → a user copies `lewm_default.yaml` for a new L0 run but doesn't know VICReg is available. **Mitigation:** `lewm_default.yaml` enables it by default, and the proposal notes the behavior change in the `Impact` section. The 1k-step baseline in AGENTS.md predates this change, so old numbers are not invalidated.

- **`projected.std(dim=0)` is ill-defined for batch size 1]** → a `B=1` batch produces `std = nan`. **Mitigation:** the existing L0 training uses `batch_size: 16` (default), so this is not a real risk. The unit test for `vicreg_loss` should use `B >= 4` to be safe and document the constraint.

- **[L1 retrain cost: ~2.5h]** → the new L0 makes the L1's existing checkpoint obsolete. **Mitigation:** this is a one-time cost, not a recurring one. The proposal documents the rebuild plan in `Impact`. The user can defer the L1 retrain if they only need the L0's geometric improvement (e.g. for visualization experiments).

- **[VICReg interacts with SIGReg]** → both shape the latent's distribution. SIGReg is a *closed-form* whitening, VICReg is a *hinge + covariance penalty*. If both push the same direction, the std term in VICReg may be redundant. **Mitigation:** the smoke test compares the per-dim std distribution with and without VICReg; if SIGReg is already enforcing the std target, the VICReg std term will be near-zero and the covariance term will dominate. This is fine.

## Migration Plan

1. **Merge this PR** with the VICReg code + tests + updated `lewm_default.yaml`
2. **Run the smoke test** (`pytest -m smoke -x`) to confirm no regressions
3. **Retrain the L0** on `data/shards/treechop_full/` with the new config — expected ~25 min with early stop, output to `checkpoints/wood_5000_vicreg/`
4. **Re-evaluate the L0's latent geometry** (PCA, ‖z‖ vs brightness) — confirm PC1 < 50% and `‖z‖` r < 0.6 with the new checkpoint
5. **(Out of scope, follow-up PR)** retrain the L1 on the new L0's latent
6. **(Out of scope, follow-up PR)** retrain the L2 on the new L1
7. **(Out of scope, follow-up PR)** regenerate `g1`, `g2`, `g3` for the new hierarchy stack
8. **(Out of scope, follow-up PR)** re-run the agent end-to-end and update the "Expected results by training size" table in `AGENTS.md`

**Rollback:** `vicreg_weight: 0.0` in any config disables VICReg, restoring the previous behavior. The merged code is bit-identical to the current behavior when VICReg is off, so rollback is a config change, not a code revert.

## Open Questions

1. **Should `std_weight` and `cov_weight` be exposed as separate config fields, or hardcoded to the paper's 25:1 ratio?** Currently we expose `vicreg_cov_weight` (default 1.0) and hardcode the variance term weight to 1.0. The paper's 25:1 ratio was calibrated for their specific loss scales; in wally the JEPA prediction loss is in the same order of magnitude as the std/cov terms, so 1:1 is a reasonable starting point. If the user finds the cov term too weak, they can set `vicreg_cov_weight: 10.0`. The proposal defaults are conservative.

2. **Should we apply VICReg on the **unprojected** encoder output as well, to shape the encoder's own representation?** The current proposal only shapes the projected output (`emb`), which is what the L1 reads. Applying to the unprojected output would add a second VICReg term and require plumbing the raw encoder output through `combined_loss`. Out of scope for this change; could be a follow-up if the projected-only VICReg proves insufficient.

3. **Should `lewm_wood_5000.yaml` (the existing 5k config that produced `checkpoints/wood_5000/checkpoint_5000.pt`) be updated to enable VICReg?** The proposal says no — users opt in by copying `lewm_default.yaml`. But this means the "5k baseline" in the AGENTS.md table continues to be the 1D-brightness L0, which may be confusing. A follow-up could add a parallel `lewm_wood_5000_vicreg.yaml` with explicit VICReg settings and update the table.
