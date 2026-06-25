# src/wally — LeWorldModel training pipeline

This subpackage owns training, planning, and the model architecture. Runs on **Windows-native Python with TheRock multi-arch PyTorch** — do not attempt training in WSL2 (librocdxg SDMA hang; see `docs/gpu-setup.md`).

## Training always uses GPU

`wally-train`, `wally-train-curriculum`, and `wally-train-hierarchy` (`src/wally/cli/train.py:54`, `src/wally/cli/train_curriculum.py:97`, `src/wally/cli/train_hierarchy.py`) default to `--device cuda` and exit with a clear error if `torch.cuda.is_available()` is False. CPU is exposed only as an explicit `--device cpu` escape hatch for a handful of fast smoke tests on tiny configs; it logs a warning. There is no environment variable that re-enables a silent CPU fallback. If training starts and the first log line is `Using device: cpu`, the active venv almost certainly has a CPU-only torch build — reinstall from the TheRock multi-arch index as shown in `docs/gpu-setup.md#windows-recommended-for-training`.

## Layout

- `models/` — ViT encoder, action embedder, causal Transformer predictor, recurrent encoder
- `data/` — WebDataset shard loading, preprocessing, dataloader, converter
- `training/` — losses, SIGReg, optimizer, scheduler, checkpoint, trainer, evaluation, curriculum, curiosity, ensemble
- `config/` — `TrainConfig`, `ModelConfig`, YAML loader
- `planner/` — CEM optimizer (with `search_space='embedding'` mode for hierarchy), latent rollout, goal-conditioned planner (accepts `target_embedding` instead of `goal_frame`), gradient MPC, subgoal detector, high-level planner, hierarchical planner
- `hierarchy/` — L1/L2/L3 JEPA world-model stack on top of the frozen L0 LeWorldModel. See `src/wally/hierarchy/AGENTS.md` for the frozen-L0 invariant, training order, and runtime streaming protocol. L0's public API is not modified; L1+ are purely additive.
- `cli/` — `wally-train`, `wally-convert`, `wally-collect`, `wally-train-curriculum`, `wally-train-hierarchy` entry points

## Data format

- **Raw shards** (`data/raw/*.tar`): per-step JPEG frames + JSON action sidecars (output of `wally-collect`; see `src/collector/AGENTS.md`)
- **Training shards** (`data/shards/*.tar`): each shard is a `.tar` containing many `.npz` files, one per **chunk** (NOT per episode). Each `.npz` holds `frames` (64, H, W, 3) uint8 and `actions` (64, 25) float32, with a contiguous 64-frame slice of one episode. Output of `wally-convert` (see `src/wally/data/converter.py` — `chunk_frames` parameter, default 64). Chunking is what makes the dataloader fast: each `.npz` is 4-7 MB compressed, so a 16-sample batch is 60-115 MB of CPU work instead of 2-5 GB. Earlier versions stored one `.npz` per full episode (2544 frames at 224x224 = 144-335 MB each), which made a 16-sample batch 2-5 GB of zip-decompression per step and starved the GPU for 30-130 s between batches. If you load a `.npz` directly expecting a full episode, you will get a 64-frame chunk — concatenate adjacent `__chunkNNN` entries from the same `episode_id` to reassemble an episode.

## Checkpoint compatibility

Pre-AdaLN checkpoints (saved before the `lewm-adaln-predictor` change) use a different model architecture (interleaved-input TransformerEncoder) and cannot be loaded by the current code. They are archived in `checkpoints/_incompatible_pre_adaln/`. New training runs start from step 0.

## Predictor architecture

The LeWorldModel predictor uses the official LeWM AdaLN-Zero design (`lucas-maes/le-wm/module.py`). Actions are passed as a conditioning sequence `c` to the `Transformer` (NOT interleaved into the latent sequence). Internal LayerNorms are `nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)` — no learnable `weight` parameter, so bf16 gradients cannot overflow. The AdaLN-Zero modulation linear (`self.modulation = nn.Linear(c_dim, 6*dim)`) is zero-initialized, so every `ConditionalBlock` is a strict identity at step 0. The previous interleaved-input design (and its `autocast(enabled=False)` fp32 wrapper) was the root cause of the bf16 NaN-gradient bug and has been removed.

## SIGReg alpha

Default `alpha: 0.1` in `configs/lewm_default.yaml` matches the LeWM paper Section 3.1 (Algorithm 1). The previous `0.01` sat at the lower edge of the paper's safe range. SIGReg is applied to the **projected** encoder output (the output of the `projector` MLP, not the raw encoder output), matching `lucas-maes/le-wm/jepa.py:39`.
