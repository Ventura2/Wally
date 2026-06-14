# lewm-adaln-predictor Specification

## Purpose
TBD - created by archiving change lewm-adaln-predictor. Update Purpose after archive.
## Requirements
### Requirement: AdaLN-Zero conditioning predictor
The system SHALL provide a causal Transformer predictor that takes a latent sequence `x` of shape `(B, T, D)` and a conditioning sequence `c` of shape `(B, T, A_emb)` and produces a predicted latent sequence of shape `(B, T, D)`. The predictor SHALL consume `c` exclusively through AdaLN-Zero modulation (see the AdaLN-Zero block requirement) and SHALL NOT interleave, concatenate, or otherwise mix `c` into the `x` tensor at the sequence level.

#### Scenario: Predict with a non-trivial action sequence
- **WHEN** the predictor is called with `x` of shape `(B, 4, 192)` and `c` of shape `(B, 4, 192)` (an already-projected action embedding)
- **THEN** the output SHALL have shape `(B, 4, 192)` and SHALL depend on both `x` and `c` (changing `c` while keeping `x` fixed SHALL change the output)

#### Scenario: Predict with zero conditioning at init
- **WHEN** the predictor is freshly initialized (all AdaLN-Zero modulation weights zeroed) and is called with any `c`
- **THEN** the output SHALL equal `x` plus the post-block LayerNorm output (i.e., the conditioning contributes nothing at step 0, identical to a strict residual identity)

#### Scenario: Causal masking prevents attending to future positions
- **WHEN** the predictor is called with `x` of length `T=8`
- **THEN** the predicted output at position `t` SHALL depend only on `x[:, :t+1, :]` and `c[:, :t+1, :]` (verified by perturbing `x[:, t+1:, :]` and asserting the output at `t` is unchanged)

### Requirement: AdaLN-Zero conditional block
The system SHALL provide a Transformer block (`ConditionalBlock`) that applies self-attention and a feed-forward network to its input `x`, with the LayerNorm scales, shifts, and residual gates produced by an AdaLN-Zero modulation linear from the conditioning signal `c`. The modulation linear SHALL be `nn.Linear(c_dim, 6 * dim)` with both weight and bias initialized to zero, so the block is a strict identity at initialization. The internal LayerNorms SHALL be `nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)`.

#### Scenario: Modulation linear starts at zero
- **WHEN** a `ConditionalBlock` is freshly constructed
- **THEN** the modulation linear's `weight` SHALL be all zeros and its `bias` SHALL be all zeros, regardless of the random seed used to construct the surrounding module

#### Scenario: Gate values are zero at init
- **WHEN** a `ConditionalBlock` is called on any input `x` and any conditioning `c`
- **THEN** the attention gate and the MLP gate SHALL both be exactly zero (verified by inspecting the chunk outputs of the modulation linear)

#### Scenario: LayerNorm has no learnable weight
- **WHEN** a `ConditionalBlock` is constructed
- **THEN** its internal `norm1`, `norm2`, and the final `Transformer.norm` SHALL be `nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)` and SHALL expose no `weight` or `bias` learnable parameters

### Requirement: Action embedder as Conv1d + MLP
The system SHALL provide an `Embedder` action-embedder that takes an action sequence of shape `(B, T, action_dim)` and produces an action-embedding sequence of shape `(B, T, emb_dim)`. The embedder SHALL consist of a `nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)` (acting as a learnable per-time-step projection) followed by a 2-layer MLP with SiLU activation, matching the `Embedder` class in the official LeWM code.

#### Scenario: Action embedder output shape
- **WHEN** an `Embedder(action_dim=25, emb_dim=192)` is called with actions of shape `(4, 16, 25)`
- **THEN** the output SHALL have shape `(4, 16, 192)`

#### Scenario: Action embedder is differentiable
- **WHEN** the embedder is called inside a training step and `loss.backward()` is invoked
- **THEN** gradients SHALL flow to both the `Conv1d.weight` and the two `nn.Linear.weight`/`bias` parameters of the embedder

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

### Requirement: Predictor signature
The predictor's `forward` method SHALL accept two positional arguments: the latent sequence `x` and the conditioning sequence `c`. It SHALL NOT accept an interleaved `(B, 2*T, D)` input. The planner rollout adapter SHALL translate `(z, action)` pairs to `(x_seq, c_seq)` before calling the predictor.

#### Scenario: Predictor rejects interleaved inputs
- **WHEN** the predictor is called with a single argument of shape `(B, 32, D)` (the old interleaved format)
- **THEN** the call SHALL raise a `TypeError` (the `forward` signature is `forward(x, c)`, not `forward(x)`)

#### Scenario: Planner adapter uses the new signature
- **WHEN** `LeWorldModelAdapter.predict(z, action)` is called with `z` of shape `(B, D)` and `action` of shape `(B, action_dim)`
- **THEN** the adapter SHALL compute `z.unsqueeze(1)` and `action.unsqueeze(1)`, run `self._model.action_embedder(...)` on the actions, and call `self._model.predictor(z_seq, a_emb)` with two arguments

### Requirement: No fp32-only autocast workaround
The predictor SHALL run in the same autocast context as the rest of the model (bf16 by default). It SHALL NOT wrap its forward in a `torch.amp.autocast(enabled=False)` context or in any other fp32-only wrapper.

#### Scenario: Predictor forward runs in bf16
- **WHEN** the model is run inside `with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):`
- **THEN** the predictor's internal ops (LayerNorm, attention, MLP) SHALL be eligible for bf16 dispatch (verified by inspecting the autocast state inside the predictor forward and asserting it is enabled)

