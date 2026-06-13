## MODIFIED Requirements

### Requirement: Transformer predictor
The system SHALL provide a causal Transformer predictor that takes a sequence of encoded latents and a separate conditioning sequence (action embeddings), and predicts the next latent representation. The predictor SHALL use causal masking to prevent attending to future tokens and SHALL consume the conditioning exclusively through AdaLN-Zero modulation (not by interleaving).

#### Scenario: Predict next latent from sequence
- **WHEN** a latent sequence `x` of shape `(B, T, D)` and a conditioning sequence `c` of shape `(B, T, A_emb)` are provided
- **THEN** the predictor SHALL output `T` predicted latents, where each prediction uses only past and current information from `x` and `c` (causal masking)

#### Scenario: Action embedding
- **WHEN** discrete or continuous actions are provided
- **THEN** actions SHALL be projected into the same embedding dimension as latents via an `Embedder` module (Conv1d + 2-layer MLP with SiLU) that is consumed by the predictor as a separate conditioning input, NOT interleaved with the latents

## MODIFIED Requirements

### Requirement: LeWorldModel assembly
The system SHALL provide a `LeWorldModel` class that composes the encoder, a `projector` MLP, an `Embedder` action-embedder, an AdaLN-Zero `ConditionalBlock`-based Transformer predictor, and a `pred_proj` MLP into a single `nn.Module`. The model SHALL expose a `forward(frames, actions)` method that returns predicted latents and encoded target latents. When `return_embeddings=True`, the third returned tensor SHALL be the projected encoder output (the SIGReg input).

#### Scenario: End-to-end forward pass
- **WHEN** `forward(frames, actions)` is called with frames of shape `(B, T, 3, 224, 224)` and actions of shape `(B, T, A_dim)`
- **THEN** the model SHALL return predicted latents and target latents suitable for computing prediction loss and SIGReg loss
