# Latent-clustering analysis

- checkpoint: `D:\Projects\Personal\artificial-intelligence\wally\checkpoints\checkpoint_100000.pt` (step 100000)
- zip: `D:\Projects\Personal\artificial-intelligence\wally\data\external\MineRLObtainIronPickaxe-v0.zip`
- raw shards: `D:\Projects\Personal\artificial-intelligence\wally\data\raw\minerl_iron`
- episodes: 5
- frames (after subsample x5): 4577
- latent dim: 192
- k-means k: 20  (inertia 14384.8)
- elapsed: 590.4s

## Cluster vs milestone agreement

| metric | observed | random baseline |
|---|---:|---:|
| NMI (geometric mean) | 0.313 | 0.006 |
| ARI (adjusted Rand) | 0.107 | -0.000 |
| Homogeneity | 0.252 | 0.005 |
| Completeness | 0.411 | 0.008 |

## Latent separation

- between-milestone centroid distance: 1.614
- within-milestone spread: 3.035
- ratio (between / within): 0.532
- linear progress R^2 (PC1..4 → milestone index): 0.382
- PC1-only R^2: 0.006

## Per-milestone purity

| milestone | n_frames | top cluster share |
|---|---:|---:|
| none | 308 | 0.27 |
| log_acquired | 855 | 0.18 |
| planks_crafted | 160 | 0.26 |
| crafting_table_placed | 79 | 0.42 |
| sticks_crafted | 692 | 0.22 |
| wooden_pickaxe | 1578 | 0.20 |
| stone_pickaxe | 486 | 0.37 |
| iron_ore_mined | 419 | 0.42 |

## Verdict

observed NMI = 0.313 (random baseline 0.006, excess +0.306)
observed ARI = 0.107 (random baseline -0.000, excess +0.107)
frames = 4577  episodes = 5

**Latents have partial milestone structure.** CEM on existing latents may work for short horizons, but the skip-k head is **likely needed** to amplify temporal abstraction for long-horizon tasks.

## Plots

- `pca_clusters.png` — PCA-2D colored by k-means cluster
- `pca_milestones.png` — PCA-2D colored by milestone (ground truth)
- `milestone_timeline.png` — stacked per-episode milestone progression
