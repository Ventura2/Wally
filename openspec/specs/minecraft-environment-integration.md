## MineStudio Environment Integration

Purpose:
- Bridge between planner and MineStudio Minecraft environment
- Enable real-time agent execution

Components:
- Env wrapper (MineStudio gym-like interface)
- Action execution loop (plan -> execute -> observe)
- Replanning strategy (replan every N steps or at fixed intervals)
- Safety bounds (action clipping, timeout handling)

Input:
- Trained LeWorldModel checkpoint
- Goal specification (frame, text, or latent)
- Environment config (seed, resolution, etc.)

Output:
- Executed trajectory (frames + actions)
- Success/failure metrics

## Open Questions
- Replanning frequency? Every step vs every N steps?
- Action interpolation between plans?
- How to handle goal specification (frame, text, or latent)?
- Episode termination criteria?
