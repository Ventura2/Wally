## MODIFIED Requirements

### Requirement: Loss
The training loss SHALL consist of a **residual prediction loss** and SIGReg regularization, combined as `L = L_pred_residual + alpha * L_sigreg`. The residual prediction loss SHALL be the MSE between the **reconstructed next-frame latent** and the true next-frame latent, where the reconstruction is `current_projected_latent + predictor(current_projected_latent, action_embedding)`. Equivalently, `L_pred_residual = F.mse_loss(emb[:, 1:] - emb[:, :-1], predicted_change)`, matching the LeWorldModel paper Algorithm 1 line 303. The SIGReg term SHALL be the closed-form Epps-Pulley statistic on random projections of the encoder's projected embeddings, transposed to `(T, B, D)`. The SIGReg term SHALL be stateless, non-negative, and SHALL NOT require a separate critic network or optimizer.

#### Scenario: Loss computation
- **WHEN** the predicted change `Δ` of shape `(B, T-1, D)`, the projected encoder embeddings `emb` of shape `(B, T, D)`, and the SIGReg input are available
- **THEN** the system SHALL compute `MSE(emb[:, 1:] - emb[:, :-1], Δ)` as the prediction loss, compute the closed-form SIGReg loss on the projected encoder embeddings (shape `(T, B, D)`), and return the weighted sum `prediction_loss + alpha * sigreg_loss`

#### Scenario: Loss is non-zero at predictor init
- **WHEN** the predictor is freshly initialized (AdaLN-Zero: `Δ = 0`) and the encoder produces non-trivial embeddings
- **THEN** the prediction loss SHALL be `MSE(emb[:, 1:] - emb[:, :-1], 0) > 0` (the variance of the frame-to-frame latent change). A prediction loss of identically zero at any logged step is a regression of this scenario.
