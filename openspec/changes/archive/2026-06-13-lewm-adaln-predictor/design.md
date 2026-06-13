## Context

The wally project's `CausalTransformerPredictor` is a custom design: it interleaves `(latent, action)` pairs into a single `(B, 2*T, D)` sequence and feeds the result to `nn.TransformerEncoder` (replaced from `nn.TransformerDecoder(memory=x)` in the previous fix attempt). The internal LayerNorms are the PyTorch defaults (`elementwise_affine=True, eps=1e-5`), and ReZero is implemented by zero-init on `self_attn.out_proj.weight` and `linear2.weight`. This combination produces non-finite gradients on the LayerNorm `weight` parameter under bf16 mixed precision when the encoder's batch statistics vary at the start of training: the grad of `LayerNorm.weight` is `(dL/dy) * centered_x` summed over the normalization dimension, and the centering subtraction can lose enough precision in bf16 to overflow. The grad guard in the trainer then skips every step, so the model never actually updates.

The official LeWM code (`lucas-maes/le-wm/module.py`) avoids this failure mode by design: its `Block` and `ConditionalBlock` use `nn.LayerNorm(elementwise_affine=False, eps=1e-6)` (no `weight` parameter to attack), and stability comes from AdaLN-Zero — a `nn.Linear(cond_dim, 6*dim)` modulation that produces `(shift, scale, gate)` for the attention and MLP branches, with the linear zero-initialized so the entire block is a strict identity at step 0. The conditioning signal `c` is the action-embedding sequence, NOT interleaved into the latent sequence. This is the canonical "Stable End-to-End JEPA" design from the paper, and the rest of the planning/agent stack in wally already assumes the model's outputs have the structure the official API produces.

The change replaces the wally predictor's internals with this official design, ports the action embedder and projector to the official shapes, and updates the planner/agent call sites. The public CLI/YAML surface is unchanged; old checkpoints are incompatible and archived.

## Goals / Non-Goals

**Goals:**
- Eliminate the NaN-gradient root cause in the predictor by removing the LayerNorm `weight` parameter and using AdaLN-Zero gates as the official ReZero.
- Match the official LeWM `module.py` architecture byte-for-byte where it matters for stability (LayerNorm eps, `elementwise_affine`, AdaLN-Zero linear init, fused SDPA attention, FeedForward design).
- Keep the CNN encoder choice from the previous fix attempt (it works on ROCm RDNA2 and produces stable BN running stats with the `@amp.custom_fwd(cast_inputs=torch.float32)` wrapper). The encoder itself is unchanged; only its output is passed through a `projector` MLP (BatchNorm1d) before the predictor sees it.
- Add `projector` (encoder-side) and `pred_proj` (predictor-side) MLPs exactly as in the official code, with BatchNorm1d as the official norm choice.
- Update planner/agent call sites to use the new predictor signature `forward(x, c)`.
- Tighten the real-data regression test so it can't pass with the grad guard silently skipping 25% of steps.

**Non-Goals:**
- Switching the encoder from CNN back to ViT (CNN is stable on this hardware, no need to re-add the ViT loading complexity).
- Re-tuning the loss / optimizer / scheduler hyperparameters (lr=1e-4, warmup=500, alpha=0.01, batch_size=16 remain).
- Implementing the recurrent encoder, Mamba-based memory, or any other architectural variants.
- Distributed / multi-GPU training.
- Resuming from old checkpoints (old checkpoints are archived; new runs start from step 0).

## Decisions

### D1. Port `Transformer` / `ConditionalBlock` / `Block` / `Attention` / `FeedForward` from `lucas-maes/le-wm/module.py:40-130` verbatim into `src/wally/models/predictor.py`.

**Rationale:** The official design is byte-for-byte the most stable configuration the LeWM team could find, validated on PushT/Cube/Reach/TwoRooms and reported in the paper. The components are simple (under 200 lines total) and have no surprising inter-component dependencies. The conditional signal `c` is the projected action-embedding sequence (output of the new `Embedder`).

**Key details from the official code (preserved as-is):**
- `nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)` for every internal norm
- `nn.Linear(dim, 6 * dim)` modulation, zero-init weight and bias → the entire block is identity at step 0
- `gate_msa = chunk_1` and `gate_mlp = chunk_5` of the modulation (6-tuple layout: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
- `modulate(x, shift, scale) = x * (1 + scale) + shift` (AdaLN modulation)
- `F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)` — fused, numerically stable
- `to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)` (no bias on the QKV projection)
- `FeedForward` uses `nn.LayerNorm(dim)` (with affine — the FFN norm is NOT modulated by AdaLN; only the pre-attention and pre-FFN norms are)
- Output `Transformer.norm = nn.LayerNorm(hidden_dim)` with default affine (not part of the AdaLN contract)

**Alternatives considered:**
- *Naive interleaving with custom LayerNorm eps=`1e-6`* — this fixes the `weight` overflow but doesn't address the `scale_msa`/`gate_msa` initialization problem (random scales can produce huge pre-softmax values). AdaLN-Zero is strictly better.
- *ReZero via zero-init on `out_proj.weight` and `linear2.weight`* — this is what the previous fix attempt did, but it doesn't make the FFN/attention contributions strictly zero when the LayerNorm output is also uninitialized (the LayerNorm produces mean-0 unit-variance output, and zero-init on the projection means zero output — same effect, less elegant).
- *Use `xFormers` or `FlashAttention` directly* — unnecessary; the official code uses `F.scaled_dot_product_attention` which dispatches to the same fused kernels where available (FlashAttention-2 on H100, Memory-Efficient Attention on RDNA2 via the TheRock backend).

### D2. Port `Embedder` (action embedder) from `lucas-maes/le-wm/module.py:174-194` verbatim.

```python
class Embedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )
    def forward(self, x):
        x = x.float()                       # (B, T, D)
        x = x.permute(0, 2, 1)              # (B, D, T)
        x = self.patch_embed(x)             # (B, smoothed_dim, T)
        x = x.permute(0, 2, 1)              # (B, T, smoothed_dim)
        x = self.embed(x)                   # (B, T, emb_dim)
        return x
```

**Rationale:** The Conv1d `kernel_size=1` is a per-time-step projection of the action across a configurable "smoothed" dimension (which we keep equal to `emb_dim` for wally's 25-dim action space, so it's effectively a learnable linear per time step followed by a 2-layer MLP). The `x.float()` cast is defensive — the input is whatever dtype the upstream encoder produced, and the embedder has no autocast context of its own.

**Alternatives considered:**
- *Keep the existing `nn.Linear(action_dim, embed_dim)` `ActionEmbedder`* — works but doesn't match the official API. The official `Embedder` has more capacity in the bottleneck (`mlp_scale * emb_dim` hidden) which helps the action signal survive the AdaLN modulation.
- *Use `nn.Embedding` (for discrete actions)* — the wally action space is a 25-dim real vector (the vpt_lib action transformer encoding), not a discrete id, so a linear-style embedder is correct.

### D3. Add `projector` and `pred_proj` as `MLP(input, hidden, output, norm_fn=BatchNorm1d, act_fn=GELU)`.

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, norm_fn=nn.LayerNorm, act_fn=nn.GELU):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )
```

**Rationale:** The projector takes the encoder's `(B, T, D)` output and projects it to the predictor's input dim. The `pred_proj` takes the predictor's output and projects it back to the target dim for the prediction loss. Both use `BatchNorm1d` (called on `(B*T, hidden)` after the first linear), which the official repo chose for its stability on single-GPU training. For wally, the CNN encoder already uses BatchNorm2d in fp32 inside autocast — BatchNorm1d in the projector follows the same pattern.

**Where they go in the data flow** (matching `le-wm/jepa.py:20-60`):
- `encode(pixels) → encoder(pixels) → projector(encoder_out) → emb`
- `act_emb = action_encoder(action)`
- `pred_emb = predictor(emb, act_emb) → pred_proj(pred_emb) → predicted`
- SIGReg is on `emb` (the projected encoder output), not on `predicted`

**Breaking signature change:** `LeWorldModel.forward(frames, actions, return_embeddings=True)` now returns `(predicted, target, emb)` where `emb` is the **projected** encoder output (i.e., what SIGReg sees), not the raw encoder output. Callers that pass `return_embeddings=False` (the planner rollout, the agent loop) get the original `(predicted, target)` tuple and are unaffected.

### D4. Drop the fp32-only `autocast(enabled=False)` workaround from the previous fix attempt.

**Rationale:** The wrapper was added because the previous predictor's `nn.LayerNorm(elementwise_affine=True, eps=1e-5)` produced NaN gradients in bf16. With the new design, there is no `LayerNorm.weight` parameter to overflow (the LayerNorms are `elementwise_affine=False`), and AdaLN-Zero gates start at 0 so the modulation cannot amplify noise. The predictor can run in bf16 end-to-end, matching the official code (which doesn't use autocast at all — it just casts the predictor to the same dtype as the encoder output).

**Verified in D5 below:** a 200-step bf16 trainer run on real Minecraft shards must complete with no NaN params and the same loss curve shape as the official paper's pre-training plots.

### D5. Test on real data, not just synthetic.

**Rationale:** The previous regression test (synthetic data, no autocast, embed_dim=64, depth=2) was too lenient: it allowed 25% of steps to be silently skipped, which is exactly what the broken predictor was doing. The new `TestLeWMRealDataStability` class in `tests/test_lewm_numerical_stability.py` is the source of truth:
- Tighten `skipped <= 25` to `skipped <= 5` (i.e., < 5% of steps may be skipped by the grad guard).
- Add a new assertion: at the end of every step (skipped or not), iterate `model.named_parameters()` and assert `torch.isfinite(p).all()` for every leaf tensor.
- Add a third test that runs 200 steps (not 100) to catch late-onset drift (e.g., BN running stats going bad after step 100+).
- Mark the new test class as `@pytest.mark.slow` (in addition to the existing `smoke` marker) so the default `pytest -m smoke` runs it.

### D6. Update the planner/agent call sites.

The `LeWorldModelAdapter.predict(z, action)` in `src/wally/planner/rollout.py:23-29` currently does:
```python
z_seq = z.unsqueeze(1)
a_seq = action.unsqueeze(1)
a_emb = self._model.action_embedder(a_seq)
interleaved = torch.cat([z_seq, a_emb], dim=1)
predicted = self._model.predictor(interleaved)
```
This becomes:
```python
z_seq = z.unsqueeze(1)                        # (B, 1, D)
a_emb = self._model.action_embedder(action.unsqueeze(1))  # (B, 1, A_emb)
predicted = self._model.predictor(z_seq, a_emb)  # (B, 1, D) — note: new signature
```
Same change in `src/wally/planner/high_level_planner.py:151` (`model._predictor(...)` call site). The `agent/loop.py` uses the `WorldModelProtocol` interface, which is `predict(z, action) → z_next` — the adapter's `predict` is the public surface, so the change is contained.

**Breaking change for any external code that calls `LeWorldModel.predictor(interleaved)` directly.** This is internal — the public model API is `LeWorldModel.forward(frames, actions)`, which keeps the same return contract for the common case (without `return_embeddings`).

## Risks / Trade-offs

- **R1: Old checkpoints are unusable.** Every old `checkpoint_*.pt` was saved by the previous predictor and has an incompatible state dict. → Mitigation: move them to `checkpoints/_incompatible_pre_adaln/` and document in `AGENTS.md` that all new runs start from step 0. The official HuggingFace checkpoints (`quentinll/lewm-*`) cannot be loaded either (they use a ViT encoder, not the wally CNN), so this is consistent — wally checkpoints are always wally-internal.
- **R2: AdaLN-Zero adds ~6 * dim * dim parameters per block.** For `dim=192, depth=4`, that's 4 * 6 * 192 * 192 = 884K extra parameters (~5% of the model). → Mitigation: negligible vs. the 15M target the paper reports; will not affect VRAM budget.
- **R3: The new test (`TestLeWMRealDataStability`, `skipped <= 5`) is stricter than the old one.** It may surface other latent bugs (e.g., in the planner rollout, in checkpoint save/load). → Mitigation: run the full smoke + slow test suite after the change; if a new failure appears, address it in this same change (it would have bitten the real run anyway).
- **R4: Recurrent encoder / `MemoryAugmentation` specs reference the predictor's old API.** The wally project also has specs for a recurrent encoder (`recurrent_encoder.py`) and memory-augmented predictors that may use the old interleaved format. → Mitigation: scope-check those specs in D6; if they need updating, file a follow-up change rather than expanding this one.
- **R5: SIGReg now operates on the **projected** embedding (192-dim), not the raw encoder output.** The output dim of the projector is the same as the encoder output (`embed_dim=192`), so the SIGReg input shape is unchanged, but the value distribution is now passed through a BatchNorm1d + GELU + Linear. The mean of the SIGReg statistic may shift, and `alpha=0.01` may need re-tuning. → Mitigation: keep `alpha=0.01` for parity with the paper; monitor the SIGReg magnitude in the first 1k steps; if it dominates the total loss, file a follow-up to tune `alpha`.
- **R6: The CNN encoder's `@amp.custom_fwd(cast_inputs=torch.float32)` was added in the previous fix to keep BN running stats in fp32.** With the new projector adding BatchNorm1d, the same fp32-in-autocast pattern must apply. → Mitigation: wrap the projector's forward in the same decorator (or apply it to the `LeWorldModel.encode` method which contains the encoder + projector).

## Migration Plan

1. Land `src/wally/models/predictor.py` rewrite (port `Transformer` / `ConditionalBlock` / `Block` / `Attention` / `FeedForward` from the official repo).
2. Land `src/wally/models/action_embedder.py` rewrite (port `Embedder`).
3. Land `src/wally/models/lewm.py` changes (add `projector` and `pred_proj` MLPs, update `forward` to return projected `emb`).
4. Land `src/wally/planner/rollout.py` and `src/wally/planner/high_level_planner.py` updates (new predictor signature, no interleaving).
5. Update `tests/test_lewm_numerical_stability.py` to use the new API; tighten the `TestLeWMRealDataStability` skip tolerance to 5% and add the param-isfinite assertion.
6. Run `pytest -m smoke -x --tb=short` — all tests pass.
7. Run `pytest -m slow -x --tb=short` (or whatever marker is registered) — the new stricter real-data test passes.
8. Run `ruff check .` and `mypy` — clean.
9. Move old `checkpoints/checkpoint_*.pt` to `checkpoints/_incompatible_pre_adaln/`.
10. Launch a fresh training run (`wally-train --config configs/lewm_default.yaml`) and monitor the first 5 minutes for finite losses and no NaN params.

**Rollback:** revert the merge commit. Old checkpoints remain in `_incompatible_pre_adaln/` and are restorable by reverting the model code. No data loss.

## Open Questions

- Should the `alpha=0.01` SIGReg weight be re-tuned now that SIGReg sees the **projected** embedding? The paper's published value is 0.1 for the closed-form SIGReg on raw encoder output, but the wally config uses 0.01. The BatchNorm1d in the projector normalizes the embedding distribution, which may change the SIGReg magnitude. → Action: keep 0.01 for parity; re-tune in a follow-up if the SIGReg dominates the loss.
- Should the `projector` and `pred_proj` share weights (tied) as in some I-JEPA variants? The official LeWM does NOT tie them. → Action: keep them separate, matching the official code.
- Should the predictor's `FeedForward` use a `nn.LayerNorm` (with affine) or `nn.LayerNorm(elementwise_affine=False)`? The official code uses default affine. → Action: keep default affine, matching the official code; the FFN norm is not part of the AdaLN contract.
