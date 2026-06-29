## ADDED Requirements

### Requirement: VICReg auxiliary loss
The L0 LeWorldModel's training loss MUST include a VICReg (Variance-Invariance-Covariance Regularization) auxiliary term that penalizes the projected encoder output for per-dim std below a target and for off-diagonal covariance, when `vicreg_weight > 0`.

The auxiliary term SHALL be `vicreg_weight * (std_loss + vicreg_cov_weight * cov_loss)`, where:
- `std_loss = mean(relu(vicreg_std_target - z.std(dim=0)))` — pushes per-dim std toward `vicreg_std_target` (default 1.0)
- `cov_loss = sum(off_diag(cov(z))^2) / D` — penalizes correlation between dims

`z` is the **projected** encoder output (the same tensor that goes into SIGReg and the JEPA predictor). `D` is the latent dimension.

When `vicreg_weight = 0` (default), the VICReg term MUST be skipped entirely so the loss is bit-identical to the current behavior.

#### Scenario: VICReg off by default
- **WHEN** a config does not set `vicreg_weight` (or sets it to `0.0`)
- **THEN** the L0's `combined_loss` returns the same value as before this change
- **AND** no VICReg computation is performed (no extra `std`/`cov` tensor allocations per step)

#### Scenario: VICReg on with default weights
- **WHEN** a config sets `vicreg_weight: 1.0` and leaves `vicreg_std_target` and `vicreg_cov_weight` at their defaults
- **THEN** the L0's `combined_loss` adds `1.0 * (std_loss + 1.0 * cov_loss)` to the existing `prediction + alpha * sigreg` sum
- **AND** the metrics dict includes a `vicreg_loss` key with the auxiliary value (before weighting)

#### Scenario: tunable std target
- **WHEN** a config sets `vicreg_std_target: 2.0`
- **THEN** the hinge term uses `gamma = 2.0` instead of the default `1.0`
- **AND** the per-dim std is pushed toward 2.0 instead of 1.0

#### Scenario: tunable covariance weight
- **WHEN** a config sets `vicreg_cov_weight: 5.0`
- **THEN** the covariance term is multiplied by 5.0 in the auxiliary sum
- **AND** the variance term is unaffected

### Requirement: Config fields for VICReg
The L0 training config MUST expose three fields that control the VICReg auxiliary loss: `vicreg_weight`, `vicreg_std_target`, `vicreg_cov_weight`. All three MUST default to values that disable VICReg (i.e. `vicreg_weight: 0.0`) and preserve the previous loss behavior.

The default config `configs/lewm_default.yaml` MUST set `vicreg_weight: 1.0` so new L0 runs opt in by default; the per-run configs (`configs/lewm_wood_*.yaml`) MUST NOT be changed by this capability.

#### Scenario: defaults preserve old behavior
- **WHEN** a config does not set any of the three new fields
- **THEN** the values default to `vicreg_weight: 0.0`, `vicreg_std_target: 1.0`, `vicreg_cov_weight: 1.0`
- **AND** the L0's training produces the same loss values as before this change

#### Scenario: lewm_default.yaml enables VICReg
- **WHEN** a new L0 run is launched with `configs/lewm_default.yaml`
- **THEN** the run uses `vicreg_weight: 1.0`
- **AND** the run writes checkpoints to the output dir as before
- **AND** the per-step log includes a `vicreg_loss` metric in addition to `prediction_loss` and `sigreg_loss`

#### Scenario: lewm_wood_*.yaml configs unchanged
- **WHEN** a config from `configs/lewm_wood_*.yaml` is used
- **THEN** the config file is not modified by this change
- **AND** the run produces the same loss values as before this change (VICReg off, because the per-run configs don't set `vicreg_weight`)

### Requirement: VICReg loss function
`src/wally/training/losses.py` MUST expose a `vicreg_loss(z, std_target, cov_weight)` function that computes the two auxiliary terms and returns a scalar tensor. The function MUST be importable and testable in isolation (no model state required).

The function signature MUST be:
```python
def vicreg_loss(z: Tensor, std_target: float = 1.0, cov_weight: float = 1.0) -> Tensor
```

The returned tensor MUST equal `mean(relu(std_target - z.std(dim=0))) + cov_weight * (off_diag(cov(z))**2).sum() / D`, where `cov(z) = (z - z.mean(0)).T @ (z - z.mean(0)) / (B - 1)`.

#### Scenario: std term forces per-dim std to target
- **WHEN** `z` is `(16, 4)` with all-ones entries
- **THEN** `z.std(dim=0)` is `(0, 0, 0, 0)` and `std_loss` is `std_target` (the hinge penalty is fully active)
- **AND** the function returns a tensor with `requires_grad=False` from the input's autograd graph perspective (the std computation must be differentiable for backprop)

#### Scenario: cov term penalizes correlated dims
- **WHEN** `z` is `(16, 4)` with column 0 equal to column 1 (perfectly correlated)
- **THEN** `cov(z)[0, 1]` is non-zero and the off-diagonal penalty is `> 0`
- **AND** `cov_weight = 0` zeroes the cov term, leaving only the std term

#### Scenario: gradients flow through both terms
- **WHEN** `z` has `requires_grad=True`
- **THEN** `vicreg_loss(z, ...).backward()` populates `z.grad` with non-zero values
- **AND** the gradients w.r.t. each dim are non-zero for both the std and cov terms (so the L0's optimizer can update the encoder's weights)

#### Scenario: batch size 1 produces NaN
- **WHEN** `z` has shape `(1, D)` for any `D >= 1`
- **THEN** the function MAY return a tensor containing NaN (this is expected — `z.std(dim=0)` is undefined for a single sample)
- **AND** the L0 training pipeline MUST guarantee `batch_size >= 4` to avoid this case (the default is `batch_size: 16`)

### Requirement: VICReg metric in training logs
When VICReg is enabled (`vicreg_weight > 0`), the per-step log line MUST include a `vicreg_loss=<value>` field alongside the existing `prediction_loss` and `sigreg_loss` fields. When disabled, the field MUST NOT appear (to keep log output identical to the pre-change baseline).

#### Scenario: log includes vicreg_loss when enabled
- **WHEN** a training run has `vicreg_weight: 1.0`
- **THEN** every log line at `log_interval` boundaries contains a `vicreg_loss` field with a finite float value
- **AND** the field is omitted entirely when `vicreg_weight: 0.0`

#### Scenario: wandb receives vicreg_loss when enabled
- **WHEN** a training run has `vicreg_weight: 1.0` and wandb is enabled
- **THEN** `wandb.log` receives a `vicreg_loss` key in the metrics dict at every `log_interval` step
- **AND** the value is finite and non-negative (the hinge + squared-cov terms are both `>= 0`)

### Requirement: Smoke test pins the VICReg behavior
A new test file `tests/test_vicreg_loss.py` MUST exist with at least the following test cases, all passing on the merged code:

1. `test_vicreg_loss_returns_finite_value_for_random_input` — call `vicreg_loss` with a `(16, 4)` random tensor; assert the result is a scalar finite float
2. `test_std_term_is_hinge_on_std` — call `vicreg_loss` with all-ones `(16, 4)`; assert the std term equals `std_target` exactly
3. `test_cov_term_zero_for_uncorrelated_input` — call `vicreg_loss` with a `(16, 4)` tensor whose columns are independent random samples; assert `cov_loss < 0.1` (sampling noise only)
4. `test_cov_term_penalizes_perfectly_correlated_columns` — construct a `(16, 4)` tensor where columns 0 and 1 are identical; assert `cov_loss > 0`
5. `test_gradients_flow_through_both_terms` — call `vicreg_loss` on a `requires_grad=True` tensor and `.backward()`; assert `z.grad` is non-zero
6. `test_combined_loss_vicreg_off_is_bit_identical` — call `combined_loss` with `vicreg_weight=0.0` and assert the returned value matches the value computed with the pre-change code path (or: assert the returned metrics dict does not contain a `vicreg_loss` key)
7. `test_combined_loss_vicreg_on_includes_metric` — call `combined_loss` with `vicreg_weight=1.0`; assert the metrics dict contains `vicreg_loss` and the total includes it

#### Scenario: smoke test suite passes
- **WHEN** `pytest -m smoke -x --tb=short` is run from the project root
- **THEN** all tests in `tests/test_vicreg_loss.py` pass
- **AND** no other smoke test changes status (the new tests are additive)
