## ADDED Requirements

### Requirement: ViT Tiny encoder
The system SHALL provide a ViT Tiny encoder (`vit_tiny_patch16_224`) that encodes RGB frames of shape `(B, 3, 224, 224)` into latent token sequences of shape `(B, N_tokens, D_embed)`. The encoder SHALL be instantiated from `timm` with optional pretrained weights.

#### Scenario: Encode a batch of RGB frames
- **WHEN** a batch of RGB frames with shape `(4, 3, 224, 224)` is passed to the encoder
- **THEN** the output SHALL be a tensor of shape `(4, 196, D_embed)` where `D_embed` is the embedding dimension (default 192)

#### Scenario: Load pretrained weights
- **WHEN** the encoder is constructed with `pretrained=True`
- **THEN** the encoder SHALL load ImageNet-pretrained weights from `timm`

### Requirement: Transformer predictor
The system SHALL provide a causal (decoder-only) Transformer predictor that takes a sequence of encoded latents and actions, and predicts the next latent representation. The predictor SHALL use causal masking to prevent attending to future tokens.

#### Scenario: Predict next latent from sequence
- **WHEN** a sequence of `(latent, action)` pairs of length `T` is provided
- **THEN** the predictor SHALL output `T` predicted latents, where each prediction uses only past and current information

#### Scenario: Action embedding
- **WHEN** discrete or continuous actions are provided
- **THEN** actions SHALL be projected into the same embedding dimension as latents via a learned linear or embedding layer

### Requirement: LeWorldModel assembly
The system SHALL provide a `LeWorldModel` class that composes the ViT encoder, action embedder, and Transformer predictor into a single `nn.Module`. The model SHALL expose a `forward(frames, actions)` method that returns predicted latents and encoded target latents.

#### Scenario: End-to-end forward pass
- **WHEN** `forward(frames, actions)` is called with frames of shape `(B, T, 3, 224, 224)` and actions of shape `(B, T, A_dim)`
- **THEN** the model SHALL return predicted latents and target latents suitable for computing prediction loss and SIGReg loss

### Requirement: Model configuration
The model architecture SHALL be configurable via a dataclass with fields for: ViT variant name, embedding dimension, Transformer depth, number of attention heads, MLP ratio, and dropout rate.

#### Scenario: Custom model configuration
- **WHEN** a `ModelConfig` is provided with `depth=6, num_heads=4, embed_dim=192`
- **THEN** the constructed model SHALL use those values for the Transformer predictor
