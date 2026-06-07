## Why

The Wally project needs a trained world model to enable downstream planning (CEM-based MPC) and goal-conditioned agents. The LeWorldModel architecture—encoding RGB frames via ViT Tiny and predicting future latents with a Transformer—is the foundation for all subsequent capabilities. No training code exists yet; this change establishes the core model training pipeline so we can begin learning Minecraft dynamics from collected trajectories.

## What Changes

- Add a Python package (`wally`) with model definitions: ViT Tiny encoder, Transformer predictor, and full LeWorldModel wrapper
- Implement training loop with prediction loss and SIGReg regularization
- Add WebDataset-based data loading for trajectory shards (RGB frames + actions)
- Add training configuration via YAML (hyperparameters, data paths, logging)
- Add evaluation utilities to visualize predictions vs. ground truth frames
- Add a CLI entry point (`wally-train`) to launch training runs

## Capabilities

### New Capabilities
- `lewm-model`: Model architecture definitions (ViT Tiny encoder, Transformer predictor, LeWorldModel assembly)
- `lewm-training-loop`: Training loop, loss functions (prediction + SIGReg), optimizer setup, checkpointing, and logging
- `lewm-data-loading`: WebDataset-based data loading pipeline for trajectory shards (RGB frames + actions)

### Modified Capabilities
- `minecraft-lewm-training`: Adds implementation-level requirements (specific hyperparameters, training schedule, data format expectations) to the existing high-level spec

## Impact

- **Code**: New `wally/` Python package with submodules for models, data, training, and config
- **Dependencies**: PyTorch, torchvision (ViT), webdataset, wandb (logging), pyyaml, einops
- **Data**: Requires collected trajectory shards in WebDataset format (from `minecraft-dataset-collector`)
- **Downstream**: Unblocks `minecraft-latent-planner` (CEM planner needs trained encoder) and `evaluation` (need trained model to evaluate)
