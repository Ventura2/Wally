## 1. Port the official LeWM module classes

- [x] 1.1 Create `src/wally/models/lewm_blocks.py` with `FeedForward`, `Attention`, `Block`, `ConditionalBlock`, and `Transformer` classes, ported verbatim from `lucas-maes/le-wm/module.py:40-130`. Use `nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)` for all internal norms, `F.scaled_dot_product_attention` with `is_causal=True` for fused attention, and zero-init on the AdaLN-Zero modulation linear.
- [x] 1.2 Create `src/wally/models/embedder.py` with the `Embedder` class ported verbatim from `lucas-maes/le-wm/module.py:174-194` (Conv1d + 2-layer MLP with SiLU).
- [x] 1.3 Create `src/wally/models/mlp.py` with the `MLP` class ported verbatim from `lucas-maes/le-wm/module.py:198-216` (Linear → optional norm → activation → Linear).

## 2. Rewrite the predictor

- [x] 2.1 Replace `src/wally/models/predictor.py` with an `ARPredictor` class that wraps the official `Transformer` with `ConditionalBlock` and a learnable `pos_embedding` of shape `(1, num_frames, input_dim)`. The forward signature is `forward(x, c) → (B, T, D)`. Drop the fp32-only `autocast(enabled=False)` wrapper. Drop the manual ReZero zero-init (AdaLN-Zero replaces it).
- [x] 2.2 Update `src/wally/models/action_embedder.py` to re-export the new `Embedder` from `embedder.py` for backward compatibility, OR delete the old file and update all call sites to import from `embedder.py`.

## 3. Update the LeWorldModel assembly

- [x] 3.1 In `src/wally/models/lewm.py`, add `self.projector = MLP(encoder_dim, hidden_dim, output_dim, norm_fn=nn.BatchNorm1d, act_fn=nn.GELU)` and `self.pred_proj = MLP(hidden_dim, hidden_dim, output_dim, norm_fn=nn.BatchNorm1d, act_fn=nn.GELU)` to the `__init__`.
- [x] 3.2 In `LeWorldModel.forward`, run the encoder as before; apply `self.projector(...)` to its output to get `emb`; pass `emb` to the predictor along with `act_emb = self.action_embedder(actions[:, :-1])` as conditioning; apply `self.pred_proj(...)` to the predictor output to get `predicted`. Return `(predicted, target, emb)` when `return_embeddings=True`, where `emb` is the projected encoder output transposed to `(T, B, D)` for SIGReg.
- [x] 3.3 Wrap the `projector`'s forward in `@torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)` so BatchNorm1d runs in fp32 inside autocast.

## 4. Update planner / agent call sites

- [x] 4.1 In `src/wally/planner/rollout.py`, change `LeWorldModelAdapter.predict(z, action)` to compute `z_seq = z.unsqueeze(1)`, `a_emb = self._model.action_embedder(action.unsqueeze(1))`, and `predicted = self._model.predictor(z_seq, a_emb).squeeze(1)`. Drop the `interleaved` and `torch.cat` lines.
- [x] 4.2 In `src/wally/planner/high_level_planner.py`, update any direct `model._predictor(...)` call to use the new `(x, c)` signature.
- [x] 4.3 In `src/wally/agent/`, audit any direct predictor / model calls; update to the new signature if any bypass the `LeWorldModelAdapter`.

## 5. Update the training loop and losses

- [x] 5.1 In `src/wally/training/losses.py`, verify `combined_loss(predicted, target, embeddings, alpha, sigreg)` matches the new projected-embedding contract (no signature change needed; the third argument is now the projected embedding transposed to `(T, B, D)`). Update the docstring to reflect this.
- [x] 5.2 In `src/wally/training/trainer.py`, verify `_training_step` still calls the model with `return_embeddings=True` and passes the result to `combined_loss`. No change needed if the loss signature is unchanged.

## 6. Update the regression tests

- [x] 6.1 In `tests/test_lewm_numerical_stability.py`, update the `TestLeWMRealDataStability` class to use the new model API. The model construction is unchanged; only the data flow through the model differs.
- [x] 6.2 In the same test class, tighten the `skipped <= 25` assertion to `skipped <= 5` and add a new assertion at the end of every step (skipped or not) that iterates `trainer.model.named_parameters()` and asserts `torch.isfinite(p).all()` for every parameter.
- [x] 6.3 In the same test class, change the test count from 100 steps to 200 steps so late-onset drift is caught.
- [x] 6.4 Add a new test `test_adaln_modulation_zero_at_init` that constructs a `ConditionalBlock` and asserts its modulation linear weight and bias are exactly zero, and that the gate chunks (positions 2 and 5 of the 6-tuple) are exactly zero for any input.

## 7. Archive old checkpoints

- [x] 7.1 Move `checkpoints/checkpoint_*.pt` to `checkpoints/_incompatible_pre_adaln/` (skip if the directory is already empty).
- [x] 7.2 Add a one-line note in `configs/lewm_default.yaml` or in `AGENTS.md` that pre-AdaLN checkpoints are incompatible with the current code.

## 8. Verify

- [x] 8.1 Run `.\.venv-windows\Scripts\python.exe -m pytest -m smoke -x --tb=short` — all existing and new tests pass.
- [x] 8.2 Run `.\.venv-windows\Scripts\python.exe -m pytest tests/test_lewm_numerical_stability.py::TestLeWMRealDataStability -v --tb=short` — the new 200-step real-data test passes with `<= 5` skipped steps and zero non-finite params.
- [x] 8.3 Run `.\.venv-windows\Scripts\python.exe -m ruff check .` — clean.
- [x] 8.4 Run `.\.venv-windows\Scripts\python.exe -m mypy` — clean.
- [x] 8.5 Launch a fresh `wally-train --config configs/lewm_default.yaml` run in the background, monitor for 5 minutes, confirm finite losses at steps 1, 10, 100, 500 with no NaN params in the saved checkpoint.

## 9. Apply the change

- [x] 9.1 Run `openspec archive lewm-adaln-predictor` to archive the change after all tasks above are complete and the new training run is confirmed stable.
