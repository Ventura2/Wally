# lewm-training-loop

## Purpose

Training loop for LeWorldModel: prediction loss, SIGReg regularization, optimizer with cosine schedule, mixed precision training, checkpointing, logging, and CLI entry point.
## Requirements
### Requirement: Prediction loss
The system SHALL compute a prediction loss as the mean squared error (MSE) between the **reconstructed next-frame latent** and the true next-frame latent. The reconstruction SHALL be `current_latent + predicted_change`, where `current_latent = projected_embeddings[:, :-1]` and `predicted_change` is the predictor's output. Equivalently, the loss SHALL be `F.mse_loss(emb[:, 1:] - emb[:, :-1], predicted_change)`, which matches the LeWorldModel paper Algorithm 1 line 303. The loss SHALL be differentiable and backpropagated through both the predictor and encoder.

#### Scenario: Compute prediction loss
- **WHEN** the predictor output (`predicted_change` of shape `(B, T-1, D)`), the projected encoder embeddings (`emb` of shape `(B, T, D)`), and the SIGReg input are provided
- **THEN** the system SHALL return `MSE(emb[:, 1:] - emb[:, :-1], predicted_change)` as a scalar loss value

#### Scenario: Loss is non-zero at predictor init
- **WHEN** the predictor is freshly initialized (AdaLN-Zero: `predicted_change = 0`) and the encoder produces non-trivial embeddings
- **THEN** the prediction loss SHALL be `MSE(emb[:, 1:] - emb[:, :-1], 0) > 0` (the variance of the frame-to-frame latent change), NOT zero. A prediction loss that is identically zero across multiple steps is a regression of this requirement.

### Requirement: SIGReg loss
The system SHALL implement SIGReg (Sketch Isotropic Gaussian Regularization) as a closed-form, stateless Epps-Pulley statistic computed on `num_proj` (default 1024) random unit-norm projections of the encoder embeddings. The statistic SHALL be non-negative, finite for any finite input embedding, and SHALL be differentiable through the encoder but not through the projection matrix. The SIGReg module SHALL expose no learnable parameters.

#### Scenario: Compute SIGReg loss on embeddings
- **WHEN** encoder embeddings of shape `(T, B, D)` are provided
- **THEN** the system SHALL return a scalar non-negative loss measuring deviation of the embedding distribution from an isotropic Gaussian

#### Scenario: SIGReg on degenerate input is finite
- **WHEN** the encoder produces an all-zero or constant embedding
- **THEN** the SIGReg loss SHALL be a finite, well-defined non-negative value (not NaN or Inf)

### Requirement: Combined training loss
The system SHALL compute a combined loss as `prediction_loss + alpha * sigreg_loss` where `alpha` is a configurable weight (default 0.1). The SIGReg loss SHALL be applied to the **projected** encoder embeddings, transposed to `(T, B, D)`, before being passed to the `SIGReg` module. The call site SHALL provide the SIGReg input in `(T, B, D)` shape directly; the SIGReg module SHALL NOT re-transpose it.

#### Scenario: Combined loss with default weight
- **WHEN** training step runs with default configuration
- **THEN** the total loss SHALL be `prediction_loss + 0.1 * sigreg_loss(proj_emb_T_B_D)` where `proj_emb_T_B_D` is the projected encoder output already transposed to `(T, B, D)` (no double-transpose at the call site)

#### Scenario: SIGReg receives (T, B, D) directly
- **WHEN** the trainer calls `combined_loss(...)` and the SIGReg module is invoked
- **THEN** the SIGReg input SHALL have shape `(T, B, D)` at the moment `SIGReg.forward` is entered (verified by patching `SIGReg.forward` to assert the shape, or by inspecting the tensor's `.shape` from inside the module)

### Requirement: Optimizer setup
The system SHALL use AdamW optimizer with configurable learning rate (default 1e-4), weight decay (default 1e-5), and a cosine annealing learning rate schedule with warmup.

#### Scenario: Optimizer with cosine schedule
- **WHEN** training begins
- **THEN** the learning rate SHALL warm up linearly for `warmup_steps` (default 1000) then decay following a cosine schedule

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

### Requirement: Logging
The system SHALL log training metrics (prediction loss, SIGReg loss, total loss, learning rate) to wandb at configurable intervals (default every 100 steps). The system SHALL also write the same metrics to stdout/stderr with `StreamHandler(sys.stdout)` and `force=True` in the logging configuration, and SHALL accept a `--log-file PATH` CLI flag that, when set, attaches a `FileHandler` writing to `PATH` in append mode. The `--log-file` handler SHALL flush after every record.

#### Scenario: Log metrics to wandb
- **WHEN** the global step reaches a log interval
- **THEN** the system SHALL log current loss values and learning rate to the active wandb run

#### Scenario: Log metrics reach disk via stdout
- **WHEN** the trainer is launched without `python -u` and without `--log-file`, but with the standard `logging.basicConfig(stream=sys.stdout, force=True)` config
- **THEN** every `logger.info` call from the trainer SHALL reach the captured stdout within one second (verified by reading the captured output after 100 logged steps and finding all 100 records)

#### Scenario: Log metrics reach disk via --log-file
- **WHEN** `wally-train --config … --log-file runs/2026-06-15.log` is launched and runs for 100 logged steps
- **THEN** the file `runs/2026-06-15.log` SHALL contain 100 metric lines, all of the form `Step N | prediction_loss=… | sigreg_loss=… | total_loss=… | lr=…`

### Requirement: Deterministic wandb run name reflecting resume state
The system SHALL initialize the wandb run with `name = f"{wandb_project}-step-{global_step}"`, where `wandb_project` is the configured project name and `global_step` is the trainer's step counter at the moment `wandb.init()` is called. This makes fresh and resumed runs identifiable in the W&B dashboard.

#### Scenario: Fresh run produces step-0 name
- **WHEN** training starts with `global_step = 0` (no resume)
- **THEN** `wandb.init()` SHALL be called with `name = "<wandb_project>-step-0"`

#### Scenario: Resumed run produces step-N name
- **WHEN** training resumes from a checkpoint at `global_step = 50000`
- **THEN** `wandb.init()` SHALL be called with `name = "<wandb_project>-step-50000"`

### Requirement: CLI entry point
The system SHALL provide a `wally-train` CLI command that accepts a path to a YAML config file and launches training.

#### Scenario: Launch training from config
- **WHEN** `wally-train --config configs/lewm.yaml` is executed
- **THEN** the system SHALL load the config, initialize model and data, and start the training loop

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

### Requirement: Anti-collapse regression scenario
A 200-step trainer run on synthetic data with the default LeWM config SHALL exhibit the following four invariants. If any invariant fails, the run is considered to have collapsed and MUST be flagged for investigation.

#### Scenario: prediction_loss is strictly positive
- **WHEN** the trainer runs 200 steps on synthetic data
- **THEN** at every logged step, `prediction_loss > 1e-6` (the loss is not identically zero or stuck at a trivial baseline)

#### Scenario: prediction_loss varies across steps
- **WHEN** the trainer runs 200 steps on synthetic data
- **THEN** `prediction_loss` SHALL NOT be constant across the run — the standard deviation across the 200 logged values SHALL be at least 1% of the mean (a perfectly flat curve is a regression of the residual-loss contract)

#### Scenario: sigreg_loss varies across steps
- **WHEN** the trainer runs 200 steps on synthetic data
- **THEN** `sigreg_loss` SHALL vary across the run — the standard deviation across the 200 logged values SHALL be at least 0.1% of the mean (a flat sigreg curve means the encoder is producing constant embeddings, which is a regression of the SIGReg contract)

#### Scenario: weight norm does not grow linearly with step
- **WHEN** 10 evenly-spaced checkpoints are sampled from a 200-step run
- **THEN** the model weight L2 norm across those 10 checkpoints SHALL NOT be perfectly linearly correlated with step number (a Pearson correlation > 0.999 between weight norm and step is a regression of the "no weight explosion" contract)

