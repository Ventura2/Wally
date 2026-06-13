# minecraft-lewm-training

## Purpose

LeWorldModel training pipeline for Minecraft: end-to-end specification covering inputs, encoder, predictor, and loss for training a world model on Minecraft gameplay trajectories.
## Requirements
### Requirement: Inputs
The LeWorldModel training pipeline SHALL accept RGB frames of shape `(B, T, 3, 224, 224)` (batch, time, channels, height, width) normalized to `[0, 1]` and action vectors of shape `(B, T, A_dim)`.

#### Scenario: Input format validation
- **WHEN** frames of shape `(4, 16, 3, 224, 224)` and actions of shape `(4, 16, 25)` are provided
- **THEN** the model SHALL accept them without error

#### Scenario: Incorrect input shape
- **WHEN** frames of shape `(4, 16, 224, 224, 3)` (channel-last) are provided
- **THEN** the system SHALL raise a validation error indicating expected channel-first format

### Requirement: Encoder
The encoder SHALL be a ViT Tiny (`vit_tiny_patch16_224`) from the `timm` library, producing latent tokens of shape `(B, 196, 192)` per frame.

#### Scenario: Encode single frame
- **WHEN** a frame of shape `(1, 3, 224, 224)` is encoded
- **THEN** the output SHALL be latent tokens of shape `(1, 196, 192)`

### Requirement: Predictor
The predictor SHALL be a causal Transformer with configurable depth (default 6), 4 attention heads, embedding dimension 192, and MLP ratio 4. It SHALL predict next-frame latents conditioned on action sequences.

#### Scenario: Predict next latent
- **WHEN** a sequence of 16 encoded latents and 16 actions are input to the predictor
- **THEN** the predictor SHALL output 16 predicted latents using causal masking

### Requirement: Loss
The training loss SHALL consist of prediction loss (MSE between predicted and target latents) and SIGReg regularization (closed-form Epps-Pulley statistic on random projections of the encoder embeddings), combined as `L = L_pred + alpha * L_sigreg`. The SIGReg term SHALL be stateless, non-negative, and SHALL NOT require a separate critic network or optimizer.

#### Scenario: Loss computation
- **WHEN** predicted and target latents and encoder embeddings are available
- **THEN** the system SHALL compute MSE prediction loss, compute the closed-form SIGReg loss on the encoder embeddings, and return the weighted sum

