## MODIFIED Requirements

### Requirement: Loss
The training loss SHALL consist of prediction loss (MSE between predicted and target latents) and SIGReg regularization (closed-form Epps-Pulley statistic on random projections of the encoder embeddings), combined as `L = L_pred + alpha * L_sigreg`. The SIGReg term SHALL be stateless, non-negative, and SHALL NOT require a separate critic network or optimizer.

#### Scenario: Loss computation
- **WHEN** predicted and target latents and encoder embeddings are available
- **THEN** the system SHALL compute MSE prediction loss, compute the closed-form SIGReg loss on the encoder embeddings, and return the weighted sum
