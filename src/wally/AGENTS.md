# src/wally — LeWorldModel training pipeline

This subpackage owns training, planning, and the model architecture. Runs on **Windows-native Python with TheRock multi-arch PyTorch** — do not attempt training in WSL2 (librocdxg SDMA hang; see `docs/gpu-setup.md`).

## Layout

- `models/` — ViT encoder, action embedder, causal Transformer predictor, recurrent encoder
- `data/` — WebDataset shard loading, preprocessing, dataloader, converter
- `training/` — losses, SIGReg, optimizer, scheduler, checkpoint, trainer, evaluation, curriculum, curiosity, ensemble
- `config/` — `TrainConfig`, `ModelConfig`, YAML loader
- `planner/` — CEM optimizer, latent rollout, goal-conditioned planner, gradient MPC, subgoal detector, high-level planner, hierarchical planner
- `cli/` — `wally-train`, `wally-convert`, `wally-collect`, `wally-train-curriculum` entry points

## Data format

- **Raw shards** (`data/raw/*.tar`): per-step JPEG frames + JSON action sidecars (output of `wally-collect`; see `src/collector/AGENTS.md`)
- **Training shards** (`data/shards/*.tar`): per-episode `.npz` files with `frames` (T,H,W,3) and `actions` (T,25) arrays (output of `wally-convert`)

## Checkpoint compatibility

Pre-AdaLN checkpoints (saved before the `lewm-adaln-predictor` change) use a different model architecture (interleaved-input TransformerEncoder) and cannot be loaded by the current code. They are archived in `checkpoints/_incompatible_pre_adaln/`. New training runs start from step 0.

## Predictor architecture

The LeWorldModel predictor uses the official LeWM AdaLN-Zero design (`lucas-maes/le-wm/module.py`). Actions are passed as a conditioning sequence `c` to the `Transformer` (NOT interleaved into the latent sequence). Internal LayerNorms are `nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)` — no learnable `weight` parameter, so bf16 gradients cannot overflow. The AdaLN-Zero modulation linear (`self.modulation = nn.Linear(c_dim, 6*dim)`) is zero-initialized, so every `ConditionalBlock` is a strict identity at step 0. The previous interleaved-input design (and its `autocast(enabled=False)` fp32 wrapper) was the root cause of the bf16 NaN-gradient bug and has been removed.

## SIGReg alpha

Default `alpha: 0.1` in `configs/lewm_default.yaml` matches the LeWM paper Section 3.1 (Algorithm 1). The previous `0.01` sat at the lower edge of the paper's safe range. SIGReg is applied to the **projected** encoder output (the output of the `projector` MLP, not the raw encoder output), matching `lucas-maes/le-wm/jepa.py:39`.
