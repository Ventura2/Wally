## Agent Evaluation & Video Recording

Purpose:
- Record agent behavior for visualization
- Quantify planning quality

Metrics:
- Latent distance to goal (final frame vs goal)
- Episode length / steps to goal
- Success rate (threshold-based)
- Action smoothness (jerk, variance)

Outputs:
- Video files (.mp4) of agent rollouts
- JSON/CSV metrics per episode
- Comparison plots (CEM vs RandomShooting)

## Open Questions
- Video codec/format?
- Real-time rendering vs post-hoc?
- Benchmark tasks (navigation, crafting, etc.)?
