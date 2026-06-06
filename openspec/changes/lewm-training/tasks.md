## 1. Project Setup

- [ ] 1.1 Create `pyproject.toml` with package metadata, dependencies (torch, timm, webdataset, wandb, pyyaml, einops), and dev dependencies (ruff, mypy, pytest)
- [ ] 1.2 Create `wally/` package directory structure: `wally/__init__.py`, `wally/models/`, `wally/data/`, `wally/training/`, `wally/config/`
- [ ] 1.3 Add `wally-train` console script entry point in `pyproject.toml`

## 2. Model Architecture

- [ ] 2.1 Implement ViT Tiny encoder wrapper in `wally/models/encoder.py` using `timm.create_model('vit_tiny_patch16_224')` with configurable pretrained flag
- [ ] 2.2 Implement action embedder in `wally/models/action_embedder.py` — linear projection from action dim to embed dim
- [ ] 2.3 Implement causal Transformer predictor in `wally/models/predictor.py` with configurable depth, heads, embed dim, MLP ratio, and dropout
- [ ] 2.4 Implement `LeWorldModel` class in `wally/models/lewm.py` composing encoder + action embedder + predictor, with `forward(frames, actions)` returning predicted and target latents
- [ ] 2.5 Add `ModelConfig` dataclass in `wally/config/model.py` with fields: vit_variant, embed_dim, depth, num_heads, mlp_ratio, dropout, action_dim

## 3. Data Loading

- [ ] 3.1 Implement WebDataset shard loader in `wally/data/dataset.py` that reads `.tar` shards with frame and action data
- [ ] 3.2 Implement sample decoder that produces `frames: (T, H, W, 3) uint8` and `actions: (T, A_dim) float32`
- [ ] 3.3 Implement frame preprocessing pipeline: uint8→float32, normalize to [0,1], resize to 224x224, transpose to channel-first `(T, 3, 224, 224)`
- [ ] 3.4 Implement sequence sampler that extracts random contiguous subsequences of configurable `seq_length` with optional skip for short trajectories
- [ ] 3.5 Implement batch collate function producing `frames: (B, T, 3, 224, 224)` and `actions: (B, T, A_dim)`
- [ ] 3.6 Implement `create_dataloader(data_dir, batch_size, num_workers, seq_length)` factory function in `wally/data/dataloader.py`

## 4. Loss Functions

- [ ] 4.1 Implement prediction loss (MSE) in `wally/training/losses.py`
- [ ] 4.2 Implement SIGReg critic MLP in `wally/training/sigreg.py` with configurable hidden dims
- [ ] 4.3 Implement SIGReg loss computation using the critic for mutual information estimation
- [ ] 4.4 Implement combined loss function: `L = L_pred + alpha * L_sigreg` with configurable alpha

## 5. Training Loop

- [ ] 5.1 Implement optimizer setup (AdamW) with configurable lr, weight decay in `wally/training/optimizer.py`
- [ ] 5.2 Implement cosine annealing LR scheduler with linear warmup in `wally/training/scheduler.py`
- [ ] 5.3 Implement training step: forward pass, combined loss, backward, gradient clipping (max norm 1.0), optimizer step
- [ ] 5.4 Add mixed precision (fp16) support via `torch.amp` with configurable `use_amp` flag
- [ ] 5.5 Implement checkpoint save/load in `wally/training/checkpoint.py` — save model state dict, optimizer state, step count, config
- [ ] 5.6 Implement wandb logging of metrics (prediction loss, SIGReg loss, total loss, lr) at configurable intervals
- [ ] 5.7 Implement main training loop in `wally/training/trainer.py` with epoch iteration, checkpointing, and logging

## 6. Configuration

- [ ] 6.1 Implement `TrainConfig` dataclass in `wally/config/training.py` with fields: lr, weight_decay, warmup_steps, max_steps, batch_size, seq_length, alpha, use_amp, checkpoint_interval, log_interval, data_dir, output_dir
- [ ] 6.2 Implement YAML config loader that parses a YAML file into `TrainConfig` and `ModelConfig`
- [ ] 6.3 Create example config file `configs/lewm_default.yaml` with paper-default hyperparameters

## 7. CLI Entry Point

- [ ] 7.1 Implement `wally/cli/train.py` with argparse accepting `--config` path
- [ ] 7.2 Wire CLI to config loading, model initialization, dataloader creation, and trainer launch

## 8. Evaluation

- [ ] 8.1 Implement evaluation utility in `wally/training/evaluate.py` that runs model on a validation batch and saves predicted vs. ground truth frame visualizations
- [ ] 8.2 Add validation loss logging to wandb at configurable intervals
