## MODIFIED Requirements

### Requirement: Prediction loss
The system SHALL compute a prediction loss as the mean squared error (MSE) between the **reconstructed next-frame latent** and the true next-frame latent. The reconstruction SHALL be `current_latent + predicted_change`, where `current_latent = projected_embeddings[:, :-1]` and `predicted_change` is the predictor's output. Equivalently, the loss SHALL be `F.mse_loss(emb[:, 1:] - emb[:, :-1], predicted_change)`, which matches the LeWorldModel paper Algorithm 1 line 303. The loss SHALL be differentiable and backpropagated through both the predictor and encoder.

#### Scenario: Compute prediction loss
- **WHEN** the predictor output (`predicted_change` of shape `(B, T-1, D)`), the projected encoder embeddings (`emb` of shape `(B, T, D)`), and the SIGReg input are provided
- **THEN** the system SHALL return `MSE(emb[:, 1:] - emb[:, :-1], predicted_change)` as a scalar loss value

#### Scenario: Loss is non-zero at predictor init
- **WHEN** the predictor is freshly initialized (AdaLN-Zero: `predicted_change = 0`) and the encoder produces non-trivial embeddings
- **THEN** the prediction loss SHALL be `MSE(emb[:, 1:] - emb[:, :-1], 0) > 0` (the variance of the frame-to-frame latent change), NOT zero. A prediction loss that is identically zero across multiple steps is a regression of this requirement.

### Requirement: Combined training loss
The system SHALL compute a combined loss as `prediction_loss + alpha * sigreg_loss` where `alpha` is a configurable weight (default 0.1). The SIGReg loss SHALL be applied to the **projected** encoder embeddings, transposed to `(T, B, D)`, before being passed to the `SIGReg` module. The call site SHALL provide the SIGReg input in `(T, B, D)` shape directly; the SIGReg module SHALL NOT re-transpose it.

#### Scenario: Combined loss with default weight
- **WHEN** training step runs with default configuration
- **THEN** the total loss SHALL be `prediction_loss + 0.1 * sigreg_loss(proj_emb_T_B_D)` where `proj_emb_T_B_D` is the projected encoder output already transposed to `(T, B, D)` (no double-transpose at the call site)

#### Scenario: SIGReg receives (T, B, D) directly
- **WHEN** the trainer calls `combined_loss(...)` and the SIGReg module is invoked
- **THEN** the SIGReg input SHALL have shape `(T, B, D)` at the moment `SIGReg.forward` is entered (verified by patching `SIGReg.forward` to assert the shape, or by inspecting the tensor's `.shape` from inside the module)

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

## ADDED Requirements

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
