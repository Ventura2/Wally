## ADDED Requirements

### Requirement: Model config is embedded in saved checkpoints

The system SHALL embed the model's architecture configuration (`encoder_type`, `vit_variant`, `embed_dim`, `depth`, `num_heads`, `mlp_ratio`, `dropout`, `action_dim`) inside the payload produced by `save_checkpoint` under the key `model_config`, alongside the existing `model_state_dict`, `optimizer_state_dict`, `global_step`, and `config` (training-config) keys. The trainer (`wally.training.trainer.Trainer`) and the `wally-train` CLI are responsible for populating it from the `ModelConfig` produced by `wally.config.loader.load_config`.

The embedded `model_config` SHALL be a JSON-serializable `dict` (no dataclass instances, no `Path` objects) so that downstream code can read it without importing the project's Pydantic / dataclass types.

#### Scenario: New checkpoints include model_config

- **WHEN** `wally-train` saves a checkpoint (whether periodic at `checkpoint_interval` steps or the final checkpoint)
- **THEN** the saved file SHALL contain a top-level `model_config` dict, and `model_config.get("encoder_type")` SHALL equal the value from the YAML config's `model.encoder_type` (e.g. `"cnn"`)

#### Scenario: Old checkpoints without model_config still load

- **WHEN** `LatentRollout._load_from_checkpoint` opens a checkpoint that does not contain a `model_config` key
- **THEN** it SHALL fall back to the existing behaviour: read `config.get("model", {})` if present, and otherwise use the constructor defaults — so checkpoints written before this change continue to load

### Requirement: LatentRollout honours the encoder_type stored in the checkpoint

`LatentRollout._load_from_checkpoint` SHALL pass the resolved `encoder_type` value to the `LeWorldModel` constructor. Resolution order: (1) `checkpoint["model_config"]["encoder_type"]`; (2) `checkpoint["config"]["model"]["encoder_type"]` (legacy); (3) the default `"vit"`.

#### Scenario: CNN checkpoint loads without state_dict mismatch

- **WHEN** `LatentRollout.from_checkpoint(path)` is called with a checkpoint saved using `encoder_type=cnn`
- **THEN** the underlying `LeWorldModel` SHALL be constructed with `encoder_type="cnn"`, `load_state_dict` SHALL succeed, and the resulting rollout SHALL operate on the CNN encoder

#### Scenario: ViT checkpoint loads without state_dict mismatch

- **WHEN** `LatentRollout.from_checkpoint(path)` is called with a checkpoint saved using `encoder_type=vit`
- **THEN** the underlying `LeWorldModel` SHALL be constructed with `encoder_type="vit"`, `load_state_dict` SHALL succeed, and the resulting rollout SHALL operate on the ViT encoder

### Requirement: LeWorldModelAdapter.encode returns a (B, Z) latent for both encoder types

`LeWorldModelAdapter.encode(frame)` SHALL return a tensor of shape `(B, embed_dim)` for both supported encoder types:

- For `encoder_type="vit"`, the encoder returns `(B, T_tokens+1, embed_dim)` and the adapter SHALL mean-pool over the token axis (`dim=1`) to produce `(B, embed_dim)`.
- For `encoder_type="cnn"`, the encoder returns `(B, embed_dim)` directly and the adapter SHALL return it unchanged (no mean across the embedding axis).

The adapter SHALL determine the encoder type at construction time from `model._is_cnn` (set by `LeWorldModel.__init__`) and SHALL NOT inspect string flags at call time.

#### Scenario: CNN encode preserves the embedding dimension

- **WHEN** `LeWorldModelAdapter.encode` is invoked on a `LeWorldModel` with `encoder_type="cnn"` and a frame of shape `(B, 3, 224, 224)`
- **THEN** the returned tensor SHALL have shape `(B, embed_dim)` where `embed_dim` is the model's configured embedding dimension

#### Scenario: ViT encode pools over the token axis

- **WHEN** `LeWorldModelAdapter.encode` is invoked on a `LeWorldModel` with `encoder_type="vit"` and a frame of shape `(B, 3, 224, 224)`
- **THEN** the returned tensor SHALL have shape `(B, embed_dim)` and SHALL equal `model.encoder(frame).mean(dim=1)` elementwise within floating-point tolerance

#### Scenario: encode output feeds predict without shape mismatch

- **WHEN** `LeWorldModelAdapter.predict(z, action)` is called with `z` produced by `encode` on the same model
- **THEN** the call SHALL NOT raise a shape-mismatch error and SHALL return a tensor of shape `(B, embed_dim)`
