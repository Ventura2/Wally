## Context

Wally is a Minecraft AI research project implementing a LeWorldModel-style pipeline. The goal is to train a latent world model on collected gameplay trajectories (RGB frames + actions), then use the trained encoder and predictor for downstream planning (CEM-based MPC) and goal-conditioned agents. No application code exists yet—this is the first implementation change.

The reference architecture (LeWorldModel paper) uses:
- A ViT encoder to compress RGB frames into latent representations
- A Transformer predictor to forecast future latents conditioned on actions
- SIGReg loss to maximize mutual information between predicted and actual latents

Existing specs define the high-level model architecture (`minecraft-lewm-training`) and data collection (`minecraft-dataset-collector`). This change establishes the training codebase.

## Goals / Non-Goals

**Goals:**
- Implement the full LeWorldModel architecture (ViT Tiny encoder + Transformer predictor)
- Implement training loop with prediction loss and SIGReg regularization
- Build a WebDataset-based data loading pipeline for trajectory shards
- Provide YAML-based configuration for hyperparameters and data paths
- Support checkpointing, logging (wandb), and basic evaluation visualization
- Establish the `wally` Python package structure for future capabilities

**Non-Goals:**
- CEM-based planning (separate change: `minecraft-latent-planner`)
- Data collection pipeline (separate capability: `minecraft-dataset-collector`)
- Distributed/multi-GPU training (future optimization)
- Fine-tuning or reinforcement learning on the world model
- Goal-conditioned agent training

## Decisions

### 1. ViT Tiny from `timm` library

**Decision**: Use `timm` (PyTorch Image Models) for the ViT Tiny encoder rather than torchvision or a custom implementation.

**Rationale**: `timm` provides a wider selection of ViT variants including `vit_tiny_patch16_224`, supports pretrained weights, and is the standard in research codebases. Torchvision's ViT support is more limited.

**Alternatives considered**:
- `torchvision.models.vit_b_16`: Fewer tiny variants, less flexible
- Custom ViT: More control but significant implementation effort for no benefit

### 2. Decoder-only Transformer for predictor

**Decision**: Use a standard decoder-only Transformer (causal masking) that takes a sequence of `(latent, action)` pairs and predicts the next latent.

**Rationale**: Matches the LeWorldModel paper's approach. Causal masking ensures the model only uses past information. Action conditioning is done by projecting actions into the same embedding space as latents and interleaving them in the sequence.

**Alternatives considered**:
- Encoder-decoder Transformer: More complex, not needed for autoregressive prediction
- Mamba/SSM: Interesting for long sequences but adds complexity and diverges from reference architecture

### 3. SIGReg loss implementation

**Decision**: Implement SIGReg (Symmetric Information Gain Regularization) as described in the LeWorldModel paper, using a learned critic network to estimate mutual information between predicted and target latents.

**Rationale**: SIGReg is critical for learning useful latent representations—it prevents the predictor from collapsing to trivial solutions. The critic is a small MLP trained alongside the main model.

### 4. Plain PyTorch training loop (no Lightning)

**Decision**: Use a plain PyTorch training loop with modular components rather than PyTorch Lightning or other frameworks.

**Rationale**: Keeps the codebase simple and transparent for research. Lightning adds abstraction that can obscure training dynamics, which matters when debugging novel losses like SIGReg. The training loop is straightforward enough (single GPU, standard optimizer) that a framework adds more overhead than value.

**Alternatives considered**:
- PyTorch Lightning: Good for standard training but adds complexity for custom loss debugging
- HuggingFace Trainer: Designed for NLP/CV fine-tuning, not custom world model training

### 5. WebDataset for data loading

**Decision**: Use WebDataset for streaming trajectory shards, with a custom collate function that assembles `(frame_sequence, action_sequence)` batches.

**Rationale**: WebDataset handles large datasets efficiently via sequential I/O on tar files, avoids loading everything into memory, and integrates naturally with PyTorch DataLoader. Matches the format planned by `minecraft-dataset-collector`.

### 6. Simple YAML config with `pyyaml`

**Decision**: Use plain YAML files loaded with `pyyaml` and a Python dataclass-based config schema, rather than Hydra or OmegaConf.

**Rationale**: Minimal dependency, easy to understand, sufficient for the number of hyperparameters involved. Hydra is powerful but overkill for a single training script.

## Risks / Trade-offs

- **[SIGReg tuning]** SIGReg has hyperparameters (critic architecture, regularization weight) that are sensitive and poorly documented. → Mitigation: Start with paper defaults, add ablation scripts early.
- **[Memory pressure]** ViT + Transformer on sequences of frames is VRAM-intensive. → Mitigation: Use gradient checkpointing, small batch sizes, and mixed precision (fp16) training.
- **[Data format coupling]** Tight coupling with `minecraft-dataset-collector` output format. → Mitigation: Define a clear data schema (frame shape, action encoding) and validate at load time.
- **[No distributed training]** Single-GPU training limits dataset size and model scale. → Mitigation: Acceptable for initial experiments with ViT Tiny; add DDP later if needed.
- **[timm dependency]** `timm` is a large dependency with frequent updates. → Mitigation: Pin version in `pyproject.toml`.
