## MODIFIED Requirements

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
