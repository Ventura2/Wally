## Why

The L0 LeWorldModel's 192-dim latent collapses to a 1-D "brightness meter": PC1 explains 84% of variance, and `‖z‖` correlates with frame brightness at r=+0.97 (per `tools/experiments/REPORT.md` §D). The L0 minimizes its prediction loss by predicting brightness, so the other 191 dims carry noise. This is a **dimensional collapse** rooted in the loss function — no amount of additional training will break it because the loss never rewards "use all dims". The downstream cost is severe:

- The L1 hierarchy trained on this latent can't distinguish "tree" from "wall" (both ≈ stable-low-brightness), so its K=32 drift stays at ~5 and it replans every tick
- The agent never picks up wood even with 18-26% of steps spent attacking, because it can't plan a multi-step "approach → face → chop" sequence
- The 1k→10k step wall-time difference is mostly wasted compute (the plateau is the L0 saturating its brightness shortcut, not learning more)

VICReg (Variance-Invariance-Covariance Regularization, Bardes et al. 2022) fixes this by adding two loss terms that explicitly force every latent dim to carry non-redundant information, without changing the JEPA prediction objective. It is the only architectural change that addresses the root cause: the L0's loss has no incentive to spread information across dims, and VICReg adds that incentive.

## What Changes

- **Add a VICReg auxiliary loss** to the L0's combined loss: `total = prediction + alpha * sigreg + vicreg_weight * vicreg(z)`, where `vicreg(z)` is a hinge on per-dim std + an off-diagonal covariance penalty applied to the **projected** encoder output (the same `z` that goes into SIGReg and the predictor)
- **Add config fields** to `TrainConfig` (`vicreg_weight`, `vicreg_std_target`, `vicreg_cov_weight`) and the L0's `lewm_default.yaml` (and the derived `lewm_wood_*.yaml` configs) — defaults that match the VICReg paper (sim_weight=25, std_weight=25, cov_weight=1) are sensible starting points
- **Retrain the L0** with the new loss on the existing `treechop_full` shards (no data change) — expected ~25 min on the 6700 XT with early stop at the new plateau
- **Re-evaluate the L0's latent geometry** (PCA, ‖z‖-vs-brightness correlation) and confirm the collapse is broken (PC1 < 50%, ‖z‖ r < 0.6) before retraining the L1
- **Retrain the L1** on the new L0's latent — this is a downstream consequence, not a direct change in this PR, but the proposal commits to the rebuild plan
- The planner (`planner/plan.py`, `planner/rollout.py`), the L1 hierarchy stack, and the agent loop are **unaffected** — the VICReg change is purely in the L0's training loss

## Capabilities

### New Capabilities
- `l0-latent-decollapse`: The L0 LeWorldModel's training loss must include a VICReg auxiliary term that penalizes per-dim std < target and per-dim covariance, applied to the projected encoder output. The loss must be tunable via config (`vicreg_weight`, `vicreg_std_target`, `vicreg_cov_weight`) and default-disabled so existing smoke tests pass.

### Modified Capabilities
- None. The L0 training is internal to `wally.cli.train`; the only externally visible behavior change is the latent's PCA profile (which no spec currently captures).

## Impact

- **Code**: `src/wally/training/losses.py` (add `vicreg_loss`), `src/wally/config/training.py` (add fields), `configs/lewm_default.yaml` + `configs/lewm_wood_*.yaml` (add defaults), `tests/` (add `test_vicreg_loss.py`)
- **Checkpoints**: the existing `checkpoints/wood_5000/checkpoint_5000.pt` becomes obsolete — the new L0 with VICReg lives at `checkpoints/wood_5000_vicreg/`. All downstream consumers (L1, L2, planner, agent) must be retrained/re-pointed. This is a one-time data-flow change, not a recurring cost
- **Training time**: ~25 min for the L0 (similar to current `lewm_wood_5000.yaml` run with early stop), plus ~2.5h for the L1 retrain on the new L0 (out of scope for this change but noted in the proposal)
- **No data change**: same `data/shards/treechop_full/` shards
- **No API change**: `wally-train` CLI is unchanged; the new fields are config-driven
- **No runtime change**: the L0 checkpoint format is unchanged; the planner, L1, and agent loop all consume the same `LeWorldModel` API
- **Risk**: the VICReg weight is a hyperparameter that may need tuning. The proposal includes a smoke test that runs 100 steps and asserts (a) the loss decreases, (b) the per-dim std distribution is non-degenerate (no dim has std ≈ 0), and (c) the off-diagonal covariance is bounded. A bad VICReg weight would either fail to decorrelate (no improvement) or destabilize training (loss diverges) — both caught by the smoke test
