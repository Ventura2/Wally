## MODIFIED Requirements

### Requirement: SIGReg loss
The system SHALL implement SIGReg (Sketch Isotropic Gaussian Regularization) as a closed-form, stateless Epps-Pulley statistic computed on `num_proj` (default 1024) random unit-norm projections of the encoder embeddings. The statistic SHALL be non-negative, finite for any finite input embedding, and SHALL be differentiable through the encoder but not through the projection matrix. The SIGReg module SHALL expose no learnable parameters.

#### Scenario: Compute SIGReg loss on embeddings
- **WHEN** encoder embeddings of shape `(T, B, D)` are provided
- **THEN** the system SHALL return a scalar non-negative loss measuring deviation of the embedding distribution from an isotropic Gaussian

#### Scenario: SIGReg on degenerate input is finite
- **WHEN** the encoder produces an all-zero or constant embedding
- **THEN** the SIGReg loss SHALL be a finite, well-defined non-negative value (not NaN or Inf)

### Requirement: Combined training loss
The system SHALL compute a combined loss as `prediction_loss + alpha * sigreg_loss` where `alpha` is a configurable weight (default 0.01). The SIGReg loss SHALL be applied to the encoder embeddings (not to predicted/target pairs), matching the LeWorldModel paper formulation.

#### Scenario: Combined loss with default weight
- **WHEN** training step runs with default configuration
- **THEN** the total loss SHALL be `prediction_loss + 0.01 * sigreg_loss`

### Requirement: Training loop
The system SHALL implement a training loop that iterates over batches, computes losses, performs gradient clipping (max norm 1.0), and updates model parameters. The loop SHALL support mixed precision (bfloat16 by default, fp16 optional) training via `torch.amp`. When AMP is active, BatchNorm layers SHALL operate in fp32 to prevent running-statistics corruption.

#### Scenario: Single training step
- **WHEN** a batch of `(frames, actions)` is provided
- **THEN** the system SHALL compute forward pass, compute combined loss, backpropagate with gradient clipping, and update parameters

#### Scenario: Mixed precision training
- **WHEN** `use_amp=True` in training config
- **THEN** the forward pass and loss computation SHALL use automatic mixed precision with bfloat16

#### Scenario: BatchNorm in autocast is fp32
- **WHEN** an encoder with BatchNorm is run inside `autocast(bfloat16)`
- **THEN** the BatchNorm forward SHALL cast its inputs to fp32 and cast its output back to the autocast dtype, keeping running statistics in fp32

#### Scenario: Non-finite loss is skipped
- **WHEN** `total_loss` is NaN or Inf at any training step
- **THEN** the system SHALL log a warning, zero gradients, skip the optimizer step, and advance `global_step` by 1

### Requirement: Checkpointing
The system SHALL save model checkpoints containing model state dict, optimizer state dict, scheduler state dict, epoch/step count, and config. Checkpoints SHALL be saved at configurable intervals (default every 5000 steps) and at the end of training.

#### Scenario: Save checkpoint
- **WHEN** the global step reaches a checkpoint interval
- **THEN** a `.pt` file SHALL be saved to the configured output directory with model weights, optimizer state, scheduler state, step count, and config

#### Scenario: Resume from checkpoint
- **WHEN** a checkpoint path is provided in config
- **THEN** the system SHALL load model weights, optimizer state, scheduler state, and resume from the saved step

#### Scenario: Resume preserves LR schedule
- **WHEN** a checkpoint containing `scheduler_state_dict` is loaded
- **THEN** the scheduler SHALL resume at the saved `last_epoch` and the next `scheduler.step()` SHALL produce the saved learning rate

#### Scenario: Resume from legacy checkpoint without scheduler state
- **WHEN** a pre-fix checkpoint is loaded that lacks `scheduler_state_dict`
- **THEN** the system SHALL initialize the scheduler with `last_epoch = global_step - 1` and log a one-time info message

## ADDED Requirements

### Requirement: NaN/Inf guard
The training loop SHALL check `torch.isfinite(total_loss).all()` after the forward pass. If the loss is not finite, the system SHALL zero the optimizer gradients, log a warning with the current step number, and skip the optimizer step (model weights are not updated for that step). The global step counter SHALL still advance to preserve logging/checkpoint cadence.

#### Scenario: All losses remain finite over a 50-step smoke run
- **WHEN** a training run executes 50 steps on synthetic data with the default config
- **THEN** every logged loss value SHALL be finite and the saved checkpoint SHALL contain no NaN/Inf parameters

### Requirement: Input sanitization
The training loop SHALL sanitize inputs by applying `torch.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)` and the same operation on `actions` immediately after moving them to the device and before the model forward pass.

#### Scenario: NaN action in batch
- **WHEN** a batch contains NaN or Inf values in `actions` or `frames`
- **THEN** the sanitization SHALL replace them with 0.0 and the forward pass SHALL complete with finite activations
