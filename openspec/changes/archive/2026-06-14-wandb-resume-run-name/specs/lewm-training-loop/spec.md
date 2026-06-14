## ADDED Requirements

### Requirement: Deterministic wandb run name reflecting resume state
The system SHALL initialize the wandb run with `name = f"{wandb_project}-step-{global_step}"`, where `wandb_project` is the configured project name and `global_step` is the trainer's step counter at the moment `wandb.init()` is called. This makes fresh and resumed runs identifiable in the W&B dashboard.

#### Scenario: Fresh run produces step-0 name
- **WHEN** training starts with `global_step = 0` (no resume)
- **THEN** `wandb.init()` SHALL be called with `name = "<wandb_project>-step-0"`

#### Scenario: Resumed run produces step-N name
- **WHEN** training resumes from a checkpoint at `global_step = 50000`
- **THEN** `wandb.init()` SHALL be called with `name = "<wandb_project>-step-50000"`
