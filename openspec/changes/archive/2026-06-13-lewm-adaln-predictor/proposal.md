## Why

The LeWorldModel predictor's current implementation interleaves latents and actions into a single `(B, 2*T, D)` sequence and feeds it to a standard `nn.TransformerDecoder` (with the `memory=x` hack that the previous change replaced with a `nn.TransformerEncoder`). It uses naive ReZero by zero-init on `out_proj.weight` and `linear2.weight`, and keeps the internal LayerNorms with their default `elementwise_affine=True`. This combination is the root cause of the persistent NaN gradients observed on the LayerNorm `weight` parameter in production runs: the grad of a `nn.LayerNorm` weight involves a sum over the normalization dimension and a multiplication by the (centered) input, and in bf16 the centering step (subtracting the running mean from an input that can swing wildly on the first few real-data batches) loses enough precision to overflow into NaN — without the input or output being NaN. The grad guard in the trainer then silently skips the step on every batch, so the model never updates.

The official LeWM paper (`lucas-maes/le-wm`, `module.py`) sidesteps this entire class of bug by (1) using `nn.LayerNorm(..., elementwise_affine=False, eps=1e-6)` — the LayerNorm has no `weight` parameter, so there is no `weight` grad to overflow; (2) using AdaLN-Zero (modulation linear zero-init, `gate_msa`/`gate_mlp` zero-init) as the official ReZero, with the action sequence passed as a **conditioning input `c`**, not interleaved into the latent sequence; and (3) using the standard `scaled_dot_product_attention` for fused, numerically stable attention. The wally predictor should match this design exactly, so that the architecture that the rest of the pipeline (planner rollout, agent loop) and the published paper rely on is the one we are actually training.

## What Changes

- **Replace the predictor with the official AdaLN-Zero design.** The `CausalTransformerPredictor` becomes a `Transformer`-with-`ConditionalBlock` stack: each block has an `Attention` (pre-LN, fused `scaled_dot_product_attention`, `is_causal=True`) followed by a `FeedForward`, with `nn.LayerNorm(elementwise_affine=False, eps=1e-6)` on each sublayer's input, and AdaLN-Zero modulation that takes the action-embedding sequence as the conditioning signal `c` and produces `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`. Both gates are zero-init so the residual branch is dead at step 0. The final output passes through a `nn.LayerNorm(elementwise_affine=False, eps=1e-6)`. **BREAKING**: the predictor's `forward(x)` signature becomes `forward(x, c)` and removes the interleaved-input assumption.
- **Rewrite the action embedder** as `Embedder(input_dim, smoothed_dim, emb_dim, mlp_scale=4)` from the official repo (Conv1d kernel_size=1 for the smoothed_dim expansion, then a 2-layer MLP with SiLU). This is what the official code uses; the linear-only `ActionEmbedder` in wally is its predecessor.
- **Wrap the encoder with a `projector` MLP and the predictor with a `pred_proj` MLP**, both of the form `MLP(input_dim, hidden_dim, output_dim, norm_fn=nn.BatchNorm1d, act_fn=nn.GELU)`. SIGReg is then applied to the **projected encoder output** (`emb`), not to the predictor's input, matching `le-wm/train.py:41`. **BREAKING**: the model's `forward(frames, actions)` returns `(predicted, target, projected_embeddings)` where the third output is now the projected latent (the input to SIGReg) rather than the raw encoder output.
- **Update the planner rollout** in `src/wally/planner/rollout.py` to call the new predictor with the action embedding as a separate `c` argument instead of interleaving. Same for `src/wally/planner/high_level_planner.py`.
- **Update the regression tests** in `tests/test_lewm_numerical_stability.py` to use the new predictor API; the `TestLeWMRealDataStability` class is the source of truth for catching this class of bug.
- **Drop the fp32-only `autocast(enabled=False)` hack** added in the previous fix attempt in `predictor.py` — it was a workaround for a bug that no longer exists once the LayerNorms have no learnable weight and AdaLN-Zero is the gate. The predictor can run in bf16 end-to-end like the official code does.
- **Mark any pre-change checkpoints as incompatible.** Checkpoints saved by the old predictor cannot be loaded by the new predictor (different state dict). New runs start from scratch; old checkpoints are kept in `checkpoints/` for archival but flagged in the config docs.

## Capabilities

### New Capabilities
- `lewm-adaln-predictor`: The LeWorldModel predictor architecture and the AdaLN-Zero conditioning pattern it relies on. Covers the block design (Attention, FeedForward, AdaLN-Zero modulation), the LayerNorm-without-affine pattern, the action-conditioning input `c`, the projector + pred_proj MLPs with BatchNorm1d, and the SIGReg-on-encoder-output contract.

### Modified Capabilities
- `lewm-model`: The "Transformer predictor" requirement changes from a sequence-of-pairs causal Transformer with interleaved latents and actions to a causal Transformer with AdaLN-Zero conditioning; the "Action embedding" requirement changes from a single linear layer to the official `Embedder` (Conv1d + 2-layer MLP with SiLU); the "LeWorldModel assembly" requirement changes so the third forward output is the projected encoder embedding (SIGReg input), not the raw encoder output.
- `lewm-numerical-stability`: The "Finite-loss training run" scenario now also asserts no NaN/Inf gradients on any parameter (previously allowed the grad guard to skip up to 25% of steps silently, which masked the predictor bug). The TestLeWMRealDataStability skipped-step tolerance is tightened to <= 5% to catch future instability.
- `lewm-training-loop`: The "Combined training loss" requirement changes so SIGReg is applied to the projected encoder output, not the raw embedding (matches the official paper's signature).

## Impact

- **Code**:
  - `src/wally/models/predictor.py` — full rewrite to AdaLN-Zero `Transformer` + `ConditionalBlock` + `FeedForward` + `Attention` (fused SDPA), with `ElementwiseAffineFalse` LayerNorms.
  - `src/wally/models/action_embedder.py` — replace `nn.Linear` with the official `Embedder(Conv1d + 2-layer MLP)`.
  - `src/wally/models/lewm.py` — add `projector` and `pred_proj` MLPs (BatchNorm1d, GELU); change `forward` signature to return `(predicted, target, projected_emb)`; remove the interleaving logic.
  - `src/wally/training/losses.py` — `combined_loss` already takes `embeddings`; verify shape contract matches the new projected embedding.
  - `src/wally/planner/rollout.py` — call `self._model.predictor(z_seq, a_emb)` instead of building `interleaved`; `LeWorldModelAdapter.predict` updated accordingly.
  - `src/wally/planner/high_level_planner.py` — update the predictor instantiation / input format if it reaches into `model._predictor` directly.
  - `src/wally/agent/` — verify the agent loop uses `LeWorldModelAdapter` (no direct predictor calls); if not, update.
  - `tests/test_lewm_numerical_stability.py` — update tests to the new API; tighten the TestLeWMRealDataStability `skipped <= 25` threshold to `skipped <= 5`; add a new assertion that no parameter has a non-finite gradient at any step.
- **Checkpoints**: old `checkpoints/checkpoint_*.pt` files are incompatible with the new architecture (different state dict). Archived under `checkpoints/_incompatible_pre_adaln/`; new runs start from step 0.
- **No user-visible CLI / YAML breaking changes** — the CLI flags and YAML keys remain identical; only the model internals change.
- **Dependencies**: no new Python packages. Requires PyTorch >= 2.0 for `F.scaled_dot_product_attention`.
