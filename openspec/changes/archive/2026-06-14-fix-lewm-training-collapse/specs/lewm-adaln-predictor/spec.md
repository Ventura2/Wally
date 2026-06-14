## MODIFIED Requirements

### Requirement: Encoder projector and predictor output projector
The system SHALL provide a `projector` MLP on the encoder output and a `pred_proj` MLP on the predictor output, both of the form `MLP(input_dim, hidden_dim, output_dim, norm_fn=nn.BatchNorm1d, act_fn=nn.GELU)`. The `projector` SHALL be applied to the encoder's `(B, T, D)` output to produce the input to the predictor. The `pred_proj` SHALL be applied to the predictor's output to produce a **predicted change** `Δ` of shape `(B, T-1, D)`. The next-frame latent is reconstructed as `next_latent = projected_embeddings[:, :-1] + Δ`; this is the formulation the LeWorldModel paper uses (Algorithm 1, line 303: `pred_loss = F.mse_loss(emb[:, 1:] - next_emb[:, :-1])`).

#### Scenario: Projector output is the SIGReg input
- **WHEN** `LeWorldModel.forward(frames, actions, return_embeddings=True)` is called
- **THEN** the third returned tensor SHALL be the output of the `projector` (not the raw encoder output), of shape `(B, T, output_dim)` matching the predictor's input dim

#### Scenario: Predicted tensor is the per-step change
- **WHEN** `LeWorldModel.forward(frames, actions)` is called
- **THEN** the first returned tensor SHALL be the predictor's output `Δ` of shape `(B, T-1, D)`, representing the frame-to-frame change in latent space (NOT the absolute next-frame latent). The caller reconstructs the next latent as `projected_embeddings[:, :-1] + Δ`.

#### Scenario: Projector uses BatchNorm1d
- **WHEN** the projector is constructed
- **THEN** its internal `nn.BatchNorm1d` layer SHALL be a learnable norm with `affine=True` (default), and SHALL cast its inputs to fp32 when run inside an `autocast(bfloat16)` context

### Requirement: SIGReg on projected encoder output
The system SHALL apply the closed-form Epps-Pulley SIGReg loss to the **projected** encoder output (i.e., the output of the `projector` MLP), not to the raw encoder output or the predictor's output. The SIGReg input SHALL be transposed to `(T, B, D)` exactly once, at the model boundary (in `LeWorldModel.forward` when `return_embeddings=True`). The SIGReg module SHALL NOT re-transpose its input — it SHALL receive `(T, B, D)` and treat the first dimension as the time axis.

#### Scenario: SIGReg sees the projected output
- **WHEN** a training step runs with `return_embeddings=True`
- **THEN** the SIGReg loss SHALL be computed on a tensor whose values are derived from the encoder's `projector` MLP output (verified by patching the projector to a constant and observing the SIGReg input change)

#### Scenario: SIGReg input is (T, B, D) at module boundary
- **WHEN** `SIGReg.forward` is entered
- **THEN** the input tensor SHALL have shape `(T, B, D)` where `T = sequence_length` and `B = batch_size` (verified by asserting `input.dim() == 3` and `input.shape[0] == T` and `input.shape[1] == B`)
