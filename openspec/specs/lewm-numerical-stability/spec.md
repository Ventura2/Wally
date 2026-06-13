# lewm-numerical-stability Specification

## Purpose
TBD - created by archiving change fix-lewm-nan-loss. Update Purpose after archive.
## Requirements
### Requirement: Closed-form SIGReg statistic
The system SHALL provide a `SIGReg` module that computes the Epps-Pulley statistic on `num_proj` random unit-norm projections of an input embedding tensor. The module SHALL be parameterized by `num_proj` (default 1024) and `knots` (default 17) and SHALL expose no learnable parameters. The forward input shape SHALL be `(T, B, D)` (time, batch, dimension) and the output SHALL be a scalar tensor.

#### Scenario: Compute SIGReg on normally distributed embeddings
- **WHEN** `SIGReg.forward` is called with embeddings drawn from a standard normal distribution
- **THEN** the output SHALL be a small positive value (close to zero, deviation from the target Gaussian)

#### Scenario: Compute SIGReg on constant embeddings
- **WHEN** `SIGReg.forward` is called with an all-zeros or constant-value embedding
- **THEN** the output SHALL be a finite, well-defined non-negative value (not NaN or Inf)

#### Scenario: Gradient flows through embedding but not projection
- **WHEN** `SIGReg.forward` is called inside a training step
- **THEN** gradients SHALL propagate to the embedding tensor that produced the input, and the projection matrix SHALL have `requires_grad=False` (it is regenerated each call)

### Requirement: Finite-loss training run
A 200-step training run on real Minecraft shards (`data/shards/chunks/*.tar`) with the default LeWM config SHALL complete with all logged losses finite, at most 5% of steps skipped by the grad guard, every parameter tensor in any saved checkpoint free of NaN/Inf, and no parameter tensor in the model state free of NaN/Inf at the end of every step (skipped or not).

#### Scenario: Smoke run produces finite loss
- **WHEN** the trainer is initialized with a `LeWorldModel(cnn encoder)`, a SIGReg module, and a real-data dataloader
- **THEN** after 200 `_training_step` invocations, `total_loss` SHALL be finite at every logged step and the grad guard SHALL have skipped no more than 10 of the 200 steps

#### Scenario: Smoke run produces NaN-free parameters at every step
- **WHEN** the trainer runs 200 steps on real data
- **THEN** after each step (including skipped steps), iterating `model.named_parameters()` SHALL yield tensors for which `torch.isfinite(p).all()` is true

#### Scenario: Smoke run produces NaN-free checkpoint
- **WHEN** the trainer saves a checkpoint at the end of a 200-step real-data run
- **THEN** every tensor in the saved state dict SHALL be free of NaN and Inf values

### Requirement: NaN recovery
If the training loss becomes non-finite at any step, the system SHALL log a warning, zero gradients, skip the optimizer step for that batch, and continue. The model weights SHALL remain unchanged for the skipped step.

#### Scenario: Forced NaN injection is recovered
- **WHEN** the model output is patched to NaN for a single batch during a 50-step smoke run
- **THEN** the system SHALL log a warning, skip that step's update, and the next batch SHALL produce a finite loss with unchanged model weights

### Requirement: Scheduler resume correctness
When a checkpoint containing `scheduler_state_dict` is loaded, the system SHALL restore the scheduler's `last_epoch` and the next `scheduler.step()` call SHALL produce the learning rate that was saved. When the checkpoint lacks `scheduler_state_dict` (legacy), the system SHALL initialize `last_epoch = global_step - 1`.

#### Scenario: LR survives a save-load round-trip
- **WHEN** the trainer saves a checkpoint at step N and then reloads it
- **THEN** `scheduler.get_last_lr()[0]` immediately after load SHALL match the LR at step N (before save)

#### Scenario: LR does not regress to warmup on resume
- **WHEN** training is resumed at global step 10000 with a 500-step warmup
- **THEN** the LR after the first resumed step SHALL NOT be near zero (warmup SHALL NOT re-run)

