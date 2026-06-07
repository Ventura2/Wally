# lewm-training-loop

## Purpose

Training loop for LeWorldModel: prediction loss, SIGReg regularization, optimizer with cosine schedule, mixed precision training, checkpointing, logging, and CLI entry point.

## Requirements

### Requirement: Prediction loss
The system SHALL compute a prediction loss as the mean squared error (MSE) between predicted latents and target latents (encoded from the next frame). This loss SHALL be differentiable and backpropagated through both the predictor and encoder.

#### Scenario: Compute prediction loss
- **WHEN** predicted latents and target latents of matching shape are provided
- **THEN** the system SHALL return a scalar MSE loss value

### Requirement: SIGReg loss
The system SHALL implement SIGReg (Symmetric Information Gain Regularization) using a learned critic MLP that estimates mutual information between predicted and target latents. The critic SHALL be trained adversarially alongside the main model.

#### Scenario: Compute SIGReg loss
- **WHEN** predicted latents and target latents are provided
- **THEN** the system SHALL compute a mutual information estimate via the critic and return a regularization loss term

#### Scenario: Critic training
- **WHEN** the training step executes
- **THEN** the critic SHALL be updated with its own optimizer to maximize mutual information estimation accuracy, while the main model minimizes the SIGReg term

### Requirement: Combined training loss
The system SHALL compute a combined loss as `prediction_loss + alpha * sigreg_loss` where `alpha` is a configurable weight (default 0.1).

#### Scenario: Combined loss with default weight
- **WHEN** training step runs with default configuration
- **THEN** the total loss SHALL be `prediction_loss + 0.1 * sigreg_loss`

### Requirement: Optimizer setup
The system SHALL use AdamW optimizer with configurable learning rate (default 1e-4), weight decay (default 1e-5), and a cosine annealing learning rate schedule with warmup.

#### Scenario: Optimizer with cosine schedule
- **WHEN** training begins
- **THEN** the learning rate SHALL warm up linearly for `warmup_steps` (default 1000) then decay following a cosine schedule

### Requirement: Training loop
The system SHALL implement a training loop that iterates over batches, computes losses, performs gradient clipping (max norm 1.0), and updates model parameters. The loop SHALL support mixed precision (fp16) training via `torch.amp`.

#### Scenario: Single training step
- **WHEN** a batch of `(frames, actions)` is provided
- **THEN** the system SHALL compute forward pass, compute combined loss, backpropagate with gradient clipping, and update parameters

#### Scenario: Mixed precision training
- **WHEN** `use_amp=True` in training config
- **THEN** the forward pass and loss computation SHALL use automatic mixed precision (fp16)

### Requirement: Checkpointing
The system SHALL save model checkpoints containing model state dict, optimizer state dict, epoch/step count, and config. Checkpoints SHALL be saved at configurable intervals (default every 5000 steps) and at the end of training.

#### Scenario: Save checkpoint
- **WHEN** the global step reaches a checkpoint interval
- **THEN** a `.pt` file SHALL be saved to the configured output directory with model weights, optimizer state, step count, and config

#### Scenario: Resume from checkpoint
- **WHEN** a checkpoint path is provided in config
- **THEN** the system SHALL load model weights, optimizer state, and resume from the saved step

### Requirement: Logging
The system SHALL log training metrics (prediction loss, SIGReg loss, total loss, learning rate) to wandb at configurable intervals (default every 100 steps).

#### Scenario: Log metrics to wandb
- **WHEN** the global step reaches a log interval
- **THEN** the system SHALL log current loss values and learning rate to the active wandb run

### Requirement: CLI entry point
The system SHALL provide a `wally-train` CLI command that accepts a path to a YAML config file and launches training.

#### Scenario: Launch training from config
- **WHEN** `wally-train --config configs/lewm.yaml` is executed
- **THEN** the system SHALL load the config, initialize model and data, and start the training loop
