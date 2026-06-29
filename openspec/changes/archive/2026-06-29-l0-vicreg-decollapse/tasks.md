## 1. Add VICReg config fields

- [x] 1.1 In `src/wally/config/training.py`, add three fields to `TrainConfig` (or a new dataclass) with defaults that disable VICReg: `vicreg_weight: float = 0.0`, `vicreg_std_target: float = 1.0`, `vicreg_cov_weight: float = 1.0`. Place them next to the existing `alpha: float = 0.1` (SIGReg) field for discoverability.
- [x] 1.2 In `src/wally/config/training.py`, add validation: `vicreg_weight >= 0`, `vicreg_std_target > 0`, `vicreg_cov_weight >= 0` (in `__post_init__` or a `field_validator`). Mirror the style of the existing `alpha` validator.
- [x] 1.3 In `configs/lewm_default.yaml`, add the three new fields under the `training:` block with `vicreg_weight: 1.0` (opt-in by default) and the other two at their defaults. Add a one-line comment explaining the field's purpose.
- [x] 1.4 Verify `configs/lewm_wood_*.yaml` files are NOT modified (the per-run configs stay on the pre-change behavior).

## 2. Implement the VICReg loss function

- [x] 2.1 In `src/wally/training/losses.py`, add the `vicreg_loss(z, std_target, cov_weight)` function as a module-level export. Use the formula from `design.md` Decision 2: `mean(relu(std_target - z.std(dim=0))) + cov_weight * (off_diag(cov(z))**2).sum() / D`. Import `torch.nn.functional as F` for `relu`.
- [x] 2.2 In `src/wally/training/losses.py`, update `combined_loss` to accept three new keyword-only parameters (`vicreg_weight`, `vicreg_std_target`, `vicreg_cov_weight`) and to add the VICReg term to the total when `vicreg_weight > 0`. When `vicreg_weight == 0`, return a metrics dict that does NOT contain a `vicreg_loss` key (preserves bit-identical log output for the default case).
- [x] 2.3 In `src/wally/training/losses.py`, update the metrics dict so that when VICReg is enabled, the `vicreg_loss` key is included (before weighting by `vicreg_weight`) and the `total_loss` includes the weighted VICReg term. Mirror the existing `sigreg_loss` key/value pattern.
- [x] 2.4 Verify the function signature matches the spec: `def vicreg_loss(z: Tensor, std_target: float = 1.0, cov_weight: float = 1.0) -> Tensor`. No `sim_weight` parameter (the prediction loss is computed by the caller, not by `vicreg_loss`).

## 3. Wire VICReg through the trainer

- [x] 3.1 In `src/wally/training/trainer.py`, locate the call site of `combined_loss` (around the `_training_step` method). Forward the three new config fields: `vicreg_weight=self.config.get("vicreg_weight", 0.0)`, `vicreg_std_target=self.config.get("vicreg_std_target", 1.0)`, `vicreg_cov_weight=self.config.get("vicreg_cov_weight", 1.0)`.
- [x] 3.2 In `src/wally/training/trainer.py`, verify the per-step log line (the one written at `log_interval` boundaries) reads `vicreg_loss` from the metrics dict only if the key is present, so the log format stays identical for the default-disabled case. Add a `vicreg_loss=<value>` field after `sigreg_loss` in the format string.
- [x] 3.3 In `src/wally/training/trainer.py`, verify that `log_metrics(metrics, step)` to wandb does not error when `metrics` has a `vicreg_loss` key. The wandb side will simply log the extra key; no code change needed beyond ensuring the key is added to `metrics` when VICReg is enabled.

## 4. Write the smoke tests

- [x] 4.1 Create `tests/test_vicreg_loss.py` with the seven test cases listed in the spec's "Smoke test pins the VICReg behavior" requirement. Use the existing `tests/test_losses.py` as a template for the test style (imports, fixtures, assertion style).
- [x] 4.2 In `tests/test_vicreg_loss.py`, add a test that asserts `vicreg_loss` is NOT called when `vicreg_weight == 0` — verify by mocking `vicreg_loss` and checking it was never called by `combined_loss` with `vicreg_weight=0.0`.
- [x] 4.3 In `tests/test_vicreg_loss.py`, add a test that asserts the metrics dict shape: with VICReg off, the dict has exactly the pre-change keys (`prediction_loss`, `sigreg_loss`, `total_loss`); with VICReg on, the dict also has `vicreg_loss`.
- [x] 4.4 Run `pytest -m smoke -x --tb=short` from the project root. All 7 new tests pass; all pre-existing smoke tests still pass.

## 5. Document the change

- [x] 5.1 In `openspec/changes/l0-vicreg-decollapse/`, verify the four artifacts (`proposal.md`, `design.md`, `specs/l0-latent-decollapse/spec.md`, `tasks.md`) all exist and are consistent. Run `openspec status --change l0-vicreg-decollapse` and confirm all `applyRequires` artifacts are `done`.
- [x] 5.2 In `AGENTS.md` (root), add a short note in the "Early stopping" section about VICReg — one paragraph explaining the latent collapse problem, the VICReg solution, and the new config fields. Mention the new `vicreg_weight` knob alongside `early_stop` and `alpha`. The note MUST point to the change directory for the full design.
- [ ] 5.3 (Optional, post-merge) Update the "Expected results by training size" table in `AGENTS.md` once the new L0 is retrained and the PCA profile is measured. This task is out of scope for this PR but is documented in the design's "Migration Plan" section.

## 6. Verify end-to-end (post-merge, not in this PR)

The following tasks are NOT in this PR — they are documented in the proposal and design but require the merged VICReg code. Track them in a follow-up issue:

- [ ] 6.1 Retrain the L0 with the new VICReg config (`configs/lewm_default.yaml`) on `data/shards/treechop_full/`, output to `checkpoints/wood_5000_vicreg/`. Expected ~25 min with early stop.
- [ ] 6.2 Re-evaluate the L0's latent geometry on the new checkpoint: run a PCA probe (similar to `tools/experiments/REPORT.md` §D) and confirm PC1 < 50% and `‖z‖` vs brightness r < 0.6.
- [ ] 6.3 (Follow-up change) Retrain the L1 on the new L0's latent using the existing `configs/hierarchy_l1_v2.yaml` (update `l0_checkpoint` to the new VICReg checkpoint). Expected ~2.5h.
- [ ] 6.4 (Follow-up change) Re-run the agent end-to-end with the new L0 + L1 stack and update the "Expected results by training size" table in `AGENTS.md`.
