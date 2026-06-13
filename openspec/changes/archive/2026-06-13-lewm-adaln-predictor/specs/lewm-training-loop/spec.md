## MODIFIED Requirements

### Requirement: Combined training loss
The system SHALL compute a combined loss as `prediction_loss + alpha * sigreg_loss` where `alpha` is a configurable weight (default 0.01). The SIGReg loss SHALL be applied to the **projected** encoder embeddings (i.e., the output of the `projector` MLP in `LeWorldModel`), matching the LeWorldModel paper formulation.

#### Scenario: Combined loss with default weight
- **WHEN** training step runs with default configuration
- **THEN** the total loss SHALL be `prediction_loss + 0.01 * sigreg_loss(proj_emb)` where `proj_emb` is the output of the `projector` MLP transposed to `(T, B, D)`
