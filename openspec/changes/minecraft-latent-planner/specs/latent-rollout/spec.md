## ADDED Requirements

### Requirement: Latent-space rollout

The system SHALL provide a `LatentRollout` module that, given an initial latent `z_0` of shape `(B, Z)` and an action sequence of shape `(B, H, A)`, returns a predicted latent trajectory of shape `(B, H+1, Z)` whose first timestep equals `z_0`, by applying the frozen `LeWorldModel` Transformer predictor for `H` steps.

The rollout SHALL be `torch.no_grad()`-compatible and SHALL support a configurable gradient policy (`"detach"` to stop gradients through predicted latents, `"straight_through"` for differentiable planning).

#### Scenario: Initial latent is preserved

- **WHEN** `LatentRollout.rollout(z_0, actions)` is called
- **THEN** `result[:, 0, :]` SHALL equal `z_0` elementwise within floating-point tolerance

#### Scenario: Batch dimension is preserved

- **WHEN** the rollout is called with a batched `z_0` of shape `(B, Z)` and a batched action sequence of shape `(B, H, A)`
- **THEN** the output SHALL have shape `(B, H+1, Z)`

#### Scenario: Detached rollout blocks gradients

- **WHEN** `LatentRollout` is constructed with `gradient_policy="detach"` and `z_0` is a leaf tensor with `requires_grad=True`
- **THEN** backpropagation through the returned trajectory SHALL NOT propagate gradients to `z_0`

### Requirement: World model consumption contract

`LatentRollout` SHALL consume a trained `LeWorldModel` (ViT-Tiny encoder + Transformer predictor) loaded from a checkpoint produced by the `lewm-training` change. The rollout SHALL NOT retrain, fine-tune, or otherwise mutate the world model parameters — they SHALL be loaded with `requires_grad=False`.

The rollout SHALL raise `ModelNotLoadedError` if instantiated without a checkpoint path or pre-loaded model object.

#### Scenario: World model parameters are frozen

- **WHEN** a trained `LeWorldModel` is loaded into a `LatentRollout` instance
- **THEN** every parameter of the underlying model SHALL have `requires_grad=False` as reported by `next(model.parameters()).requires_grad`

#### Scenario: Missing checkpoint raises

- **WHEN** `LatentRollout(checkpoint_path=None, model=None)` is constructed
- **THEN** construction SHALL raise `ModelNotLoadedError`
