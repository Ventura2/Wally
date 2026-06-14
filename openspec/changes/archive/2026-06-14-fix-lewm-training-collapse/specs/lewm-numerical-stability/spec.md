## MODIFIED Requirements

### Requirement: Closed-form SIGReg statistic
The system SHALL provide a `SIGReg` module that computes the Epps-Pulley statistic on `num_proj` random unit-norm projections of an input embedding tensor. The module SHALL be parameterized by `num_proj` (default 1024) and `knots` (default 17) and SHALL expose no learnable parameters. The forward input shape SHALL be `(T, B, D)` (time, batch, dimension) and the output SHALL be a scalar tensor. The module SHALL assert `input.dim() == 3` on entry; it SHALL NOT re-transpose its input.

#### Scenario: Compute SIGReg on normally distributed embeddings
- **WHEN** `SIGReg.forward` is called with embeddings drawn from a standard normal distribution of shape `(T, B, D)`
- **THEN** the output SHALL be a small positive value (close to zero, deviation from the target Gaussian)

#### Scenario: Compute SIGReg on constant embeddings
- **WHEN** `SIGReg.forward` is called with an all-zeros or constant-value embedding of shape `(T, B, D)`
- **THEN** the output SHALL be a finite, well-defined non-negative value (not NaN or Inf)

#### Scenario: Gradient flows through embedding but not projection
- **WHEN** `SIGReg.forward` is called inside a training step
- **THEN** gradients SHALL propagate to the embedding tensor that produced the input, and the projection matrix SHALL have `requires_grad=False` (it is regenerated each call)

#### Scenario: Input shape is asserted as 3D
- **WHEN** `SIGReg.forward` is called with a tensor of `dim() != 3`
- **THEN** the call SHALL raise a `ValueError` (or `AssertionError`) with a message indicating the expected `(T, B, D)` shape. A silent re-transpose is a regression of this contract.

### Requirement: Finite-loss training run
A 200-step training run on real Minecraft shards (`data/shards/chunks/*.tar`) with the default LeWM config SHALL complete with all logged losses finite, at most 5% of steps skipped by the grad guard, every parameter tensor in any saved checkpoint free of NaN/Inf, and no parameter tensor in the model state free of NaN/Inf at the end of every step (skipped or not). Additionally, the run SHALL NOT exhibit the "training collapse" failure mode: `prediction_loss` MUST be strictly positive at every logged step, MUST vary across steps, and `sigreg_loss` MUST vary across steps. The model weight L2 norm MUST NOT grow linearly with step (Pearson correlation between weight norm and step < 0.999 across 10 evenly-spaced checkpoints).

#### Scenario: Smoke run produces finite loss
- **WHEN** the trainer is initialized with a `LeWorldModel(cnn encoder)`, a SIGReg module, and a real-data dataloader
- **THEN** after 200 `_training_step` invocations, `total_loss` SHALL be finite at every logged step and the grad guard SHALL have skipped no more than 10 of the 200 steps

#### Scenario: Smoke run produces NaN-free parameters at every step
- **WHEN** the trainer runs 200 steps on real data
- **THEN** after each step (including skipped steps), iterating `model.named_parameters()` SHALL yield tensors for which `torch.isfinite(p).all()` is true

#### Scenario: Smoke run produces NaN-free checkpoint
- **WHEN** the trainer saves a checkpoint at the end of a 200-step real-data run
- **THEN** every tensor in the saved state dict SHALL be free of NaN and Inf values

#### Scenario: Smoke run does not collapse (anti-regression)
- **WHEN** the trainer runs 200 steps with the default config
- **THEN** the run SHALL satisfy the four invariants from `lewm-training-loop` requirement "Anti-collapse regression scenario": `prediction_loss > 1e-6` at every step, `prediction_loss` std/mean ≥ 1%, `sigreg_loss` std/mean ≥ 0.1%, and the Pearson correlation between model weight L2 norm and step number (across 10 evenly-spaced checkpoints) SHALL be < 0.999
