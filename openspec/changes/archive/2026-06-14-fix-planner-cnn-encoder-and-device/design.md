## Context

The planner integration has three concrete defects discovered while building `tools/eval_goals.py`:

1. **Encoder type is dropped on checkpoint load.** `LatentRollout._load_from_checkpoint` in `src/wally/planner/rollout.py` reads `vit_variant`, `embed_dim`, `depth`, `num_heads`, `mlp_ratio`, `dropout`, and `action_dim` from the checkpoint config but **not** `encoder_type`. It then constructs `LeWorldModel(..., pretrained=False)` which defaults to `encoder_type="vit"`. Every checkpoint saved with `encoder_type=cnn` (the project's default in `configs/lewm_default.yaml`) therefore fails `load_state_dict` with missing/extra keys.

2. **CNN encoder is shape-collapsed by the planner adapter.** `LeWorldModelAdapter.encode` does `tokens = self._model.encoder(frame); return tokens.mean(dim=1)`. For ViT, the encoder output is `(B, T_tokens+1, D)` and pooling over the token axis yields `(B, D)` — correct. For the CNN encoder (`SimpleCNNEncoder`), the output is already `(B, D)`, so `mean(dim=1)` averages over the embedding dimension and returns `(B, 1)`. Subsequent `predict(z)` calls then fail with a shape mismatch on the predictor.

3. **CEM samples live on CPU regardless of the model device.** `CEMOptimizer.optimize` creates candidates via `torch.randn(...)` (CPU by default) and only later `clamp`s them. Planners pass the cost function a world model on `cuda` plus a `z_0` on `cuda`; the resulting cost function call does `z_0_exp.expand(...)` (CUDA) × `actions` (CPU), and the rollout raises `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same`.

Stakeholders: the `wally-plan`, `wally-plan-hierarchical`, `wally-play`, and `wally-deploy` CLIs all consume `LatentRollout`/`GoalConditionedPlanner`/`HierarchicalPlanner`/`GradientMPC`, so this fix unblocks every downstream user. The `wally-train` pipeline is unaffected (training uses the model directly, not the planner adapter).

## Goals / Non-Goals

**Goals**

- Make `LatentRollout.from_checkpoint` load CNN-encoder checkpoints with no `state_dict` mismatch.
- Make `LeWorldModelAdapter.encode` return a `(B, Z)` latent for both ViT and CNN encoders.
- Make planners (CEM-based, gradient-MPC, hierarchical) work on `cuda` without callers having to manually move candidates to device.
- Keep the existing CPU + ViT path byte-for-byte identical so no regression is possible there.
- Add tests that exercise the previously-broken paths.

**Non-Goals**

- No new action spaces, no new planner algorithms, no model architecture changes.
- No new CLI flags on `wally-plan` etc. The fix is transparent to the CLI surface.
- No changes to checkpoint *contents* beyond embedding the model config (a small, backward-compatible addition — see Decision 2).

## Decisions

### Decision 1: Fix the adapter to branch on the actual encoder type, not on a flag string

The model already sets `self._is_cnn` in `LeWorldModel.__init__`. The adapter can read that attribute and either return `encoder(frame)` directly (CNN case) or `encoder(frame).mean(dim=1)` (ViT case). This is strictly more correct than the current code and is the smallest possible diff.

**Alternative considered:** re-architect the encoder to always return `(B, T_tokens, D)` (CNN returns `T_tokens=1`) so the adapter never needs to know. Rejected: it touches the model + every existing call site, and the project has no test that exercises "uniform encoder output shape". Branching in the adapter is the surgical fix.

### Decision 2: Embed the model config in the checkpoint so `LatentRollout.from_checkpoint` is self-sufficient

Today the saved checkpoint's `config` field is the *training* config (lr, weight_decay, …) — the *model* config (embed_dim, depth, num_heads, mlp_ratio, action_dim, encoder_type, vit_variant) is reconstructed implicitly from the default `LeWorldModel(...)` constructor, which is why the encoder_type is lost. Two ways to fix:

- **A) Save it explicitly.** Update `Trainer.__init__` (or the `wally.cli.train` script) to pass the `ModelConfig` through and have `save_checkpoint` store it under `checkpoint["model_config"]`. `LatentRollout._load_from_checkpoint` then reads `checkpoint.get("model_config")` (with the existing `ckpt["config"].get("model", {})` fallback for old checkpoints).
- **B) Make the caller pass the model config in.** `LatentRollout.from_checkpoint(path, model_config=...)` — every CLI / tool that loads a checkpoint is updated to read the YAML and pass it in.

**Choice: A.** It removes the failure mode entirely (old tools stop needing to know about YAML configs) and is the same number of changed lines as B. Backward compatibility: if `model_config` is missing, fall back to the current behaviour so pre-existing checkpoints still load.

**Alternative considered:** storing the full `LeWorldModel` instead of its state dict (e.g. via `torch.save({"model": model, ...})`). Rejected: it pickles the whole object graph (BatchNorm running stats, Python class identity, etc.) and is harder to load cross-version.

### Decision 3: Make `CEMOptimizer` device-aware, plumb device through planners

`CEMOptimizer.optimize` gains an optional `device: torch.device | str | None = None` argument. When set, candidates are created on that device via `torch.randn(shape, generator=rng, device=device)`. When unset, the historical CPU behaviour is preserved.

Each planner (`GoalConditionedPlanner`, `GradientMPC`, `HierarchicalPlanner`, `HighLevelPlanner`) already has a `self._device`; it just doesn't pass it down. The change is to forward `device=self._device` into the `cem.optimize` call site (one line per planner). No other planner logic changes.

**Alternative considered:** a single global "move all cost-fn tensors to CPU before/after" wrapper. Rejected: it would mask future device bugs and is the exact pattern that has been silently working around this issue. Making the contract explicit is the better fix.

### Decision 4: Keep the new CEM `device` parameter optional

Backwards compatibility: every existing test calls `cem.optimize(...)` without a device; we don't want a wave of test churn. The new default is `None` → CPU. Anyone who wants CUDA explicitly passes `device="cuda"`.

## Risks / Trade-offs

- **[Risk]** Adding a field to the saved checkpoint (Decision 2) could in principle break a downstream consumer that reads the checkpoint by hand and validates its schema. → **Mitigation:** the new field is additive and namespaced (`model_config`); old consumers ignore unknown keys, and old checkpoints remain loadable via the fallback.
- **[Risk]** Branching in the adapter (Decision 1) means the ViT-vs-CNN contract is enforced by code, not by type. → **Mitigation:** add a test that loads a CNN checkpoint and a ViT checkpoint (if any pre-AdaLN ViT checkpoint still exists in `checkpoints/_incompatible_pre_adaln/`) and checks that `encode` returns `(B, embed_dim)` for both.
- **[Risk]** Forcing the CEM device parameter could regress CPU-only runs if a planner passes `device="cpu"` and the user's `cost_fn` happens to mix devices. → **Mitigation:** the planner already pins its `self._device`; if a cost fn breaks that, it was already broken — we are not making it worse. Add a smoke test that exercises a CPU-only `GoalConditionedPlanner.plan_to_latent` to lock in the behaviour.
- **[Trade-off]** A small amount of code in `tools/eval_goals.py` becomes redundant once the project is fixed. → **Mitigation:** as part of the implementation, strip the inline `_load_world_model`, `_CNNCompatibleAdapter`, and the `--device cpu` default. The script then reverts to the natural "let the planner handle it" call style.

## Migration Plan

1. Land the code changes (Trainer + checkpoint + rollout + adapter + CEM + planners).
2. Land the new tests; the existing suite must remain green.
3. Strip the workarounds in `tools/eval_goals.py` and re-run its smoke test (`--mode mock`).
4. Re-run the existing `tests/test_latent_rollout.py`, `tests/test_cem.py`, `tests/test_goal_conditioned_planner.py`, `tests/test_hierarchical_planner.py`, `tests/test_gradient_mpc.py`, `tests/test_planner_cli.py` to confirm no regression.
5. Manual: run `python tools/eval_goals.py --checkpoints 'checkpoints/checkpoint_*.pt' --num-checkpoints 2 --mode world_model --episodes 1` against the real checkpoints to confirm end-to-end.

No database, no infra, no model retraining required. Rollback is a `git revert` of the change.

## Open Questions

- Should the planner also accept a `model_config` argument as a fallback for *old* checkpoints that don't have `model_config` embedded? (Yes, with a deprecation warning, if it's cheap. Resolved at implementation time.)
- Should `LeWorldModelAdapter.encode` be moved to `LeWorldModel.encode` itself so the model owns its encoder contract? (Out of scope for this change — would be a separate refactor; tracked here for awareness.)
