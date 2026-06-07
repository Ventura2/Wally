## Hierarchical Latent Planning

Purpose:
- Enable long-horizon task planning (e.g., "craft diamond sword")
- Decompose complex goals into subgoals
- Plan at multiple temporal scales
- Address partial observability in Minecraft environments

Architecture:
- High-level LeWorldModel: Predicts intermediate latent subgoals (5-10 macro steps)
- Low-level LeWorldModel: Plans primitive actions between subgoals (8-16 steps)
- Memory augmentation: LSTM/Transformer-XL for tracking observation history
- Gradient-based MPC: Refine action sequences at low level with differentiable planning

Components:
- Subgoal generator (THICK/C-RSSM for context-change detection)
- High-level planner (CEM over temporally abstract transitions)
- Low-level planner (gradient descent on primitive actions)
- Recurrent encoder (LSTM/Transformer-XL for history context)
- Intrinsic curiosity module (prediction error as exploration reward)
- Safety-critic (constraint checking and uncertainty estimation)

Input:
- Current frame + goal frame (RGB images)
- Optional: text description of goal (future)
- Optional: subgoal sequence from high-level planner

Output:
- Action sequence to reach goal (primitive actions)
- Intermediate subgoals (for monitoring and debugging)
- Confidence/uncertainty estimates for plan quality

## Requirements

### High-Level Planning
The system SHALL detect context-change points in trajectories using THICK or C-RSSM algorithms.
The system SHALL train a high-level LeWorldModel on temporally abstract transitions between context-change points.
The system SHALL plan sequences of latent subgoals toward the final goal using CEM optimization.
The system SHALL support planning horizons of 5-10 macro steps at the high level.

#### Scenario: Subgoal detection
- **WHEN** agent observes trajectory with distinct phases (e.g., mining → crafting → building)
- **THEN** system identifies context-change points and creates abstract transition dataset

#### Scenario: High-level planning
- **WHEN** agent receives goal (e.g., "craft diamond sword")
- **THEN** high-level planner generates sequence of latent subgoals: [mine_wood, craft_table, mine_iron, smelt_iron, mine_diamond, craft_sword]

#### Scenario: Subgoal execution
- **WHEN** high-level planner produces subgoal sequence
- **THEN** low-level planner executes each subgoal sequentially, replanning if subgoal not reached within timeout

### Low-Level Planning
The system SHALL use existing LeWorldModel for primitive action planning between subgoals.
The system SHALL implement gradient-based MPC for differentiable action optimization.
The system SHALL warm-start optimization with auxiliary value/policy network (Dreamer-style).
The system SHALL support planning horizons of 8-16 primitive steps at the low level.

#### Scenario: Gradient-based planning
- **WHEN** agent needs to reach next subgoal
- **THEN** system uses gradient descent on action sequence to minimize distance to subgoal latent

#### Scenario: Warm-starting
- **WHEN** agent has auxiliary value/policy network
- **THEN** system uses network output as initial mean for CEM/gradient optimization

#### Scenario: Replanning on failure
- **WHEN** low-level planner fails to reach subgoal within timeout
- **THEN** system requests new subgoal from high-level planner or adjusts strategy

### Memory Augmentation
The system SHALL add LSTM or Transformer-XL module to encoder for tracking observation history.
The system SHALL maintain recurrent hidden state across planning steps.
The system SHALL address partial observability by incorporating recent history into latent representation.
The system SHALL support configurable memory length (e.g., last 10-50 frames).

#### Scenario: Recurrent encoding
- **WHEN** agent observes new frame
- **THEN** encoder combines current frame with LSTM hidden state to produce context-aware latent

#### Scenario: Partial observability handling
- **WHEN** agent needs to remember past observations (e.g., location of resources)
- **THEN** recurrent encoder maintains this information in hidden state

#### Scenario: Memory reset
- **WHEN** agent completes task or dies
- **THEN** system resets LSTM hidden state to initial values

### Curriculum Training
The system SHALL train world models on increasing horizon (short clips → full complexity).
The system SHALL use shaped costs for planning (pseudo-rewards for progress toward subgoals).
The system SHALL support curriculum stages: 8-step → 16-step → 32-step → full horizon.

#### Scenario: Progressive training
- **WHEN** training hierarchical world model
- **THEN** system starts with 8-step sequences, gradually increases to 32+ steps

#### Scenario: Shaped costs
- **WHEN** planning toward subgoal
- **THEN** system includes intermediate rewards for proximity to subgoal (not just final goal)

#### Scenario: Curriculum progression
- **WHEN** model achieves target performance on current horizon
- **THEN** system increases horizon and continues training

### Exploration
The system SHALL use curiosity-based data collection (prediction error as intrinsic reward).
The system SHALL ensure diverse coverage of state space for world model training.
The system SHALL balance exploration (novel states) vs. exploitation (goal-directed behavior).

#### Scenario: Curiosity-driven exploration
- **WHEN** collecting training data
- **THEN** system prioritizes states with high prediction error (novel situations)

#### Scenario: Diverse dataset
- **WHEN** training world model
- **THEN** dataset contains varied trajectories covering different scenarios

#### Scenario: Exploration-exploitation tradeoff
- **WHEN** agent is deployed
- **THEN** system balances exploring new areas vs. pursuing goal efficiently

### Safety and Robustness
The system SHALL check constraints in planning (discard trajectories that violate safety bounds).
The system SHALL use ensemble or dropout in world model to gauge uncertainty.
The system SHALL avoid overconfident wrong plans by considering uncertainty estimates.
The system SHALL support safety-critic signals for fine-tuning (e.g., collision avoidance).

#### Scenario: Constraint checking
- **WHEN** planner generates candidate trajectory
- **THEN** system discards trajectories that violate safety bounds (e.g., walking into lava)

#### Scenario: Uncertainty estimation
- **WHEN** world model predicts next state
- **THEN** system provides uncertainty estimate (via ensemble variance or dropout)

#### Scenario: Safe planning
- **WHEN** planning under uncertainty
- **THEN** system prefers conservative plans with high confidence

## Implementation Approach

### Phase 1: Subgoal Detection
1. Collect diverse trajectories with context changes
2. Implement THICK or C-RSSM for context-change detection
3. Extract abstract transitions between context-change points
4. Validate subgoal quality on simple tasks

### Phase 2: High-Level World Model
1. Train high-level LeWorldModel on abstract transitions
2. Implement CEM planner for subgoal sequences
3. Test on 2-3 step tasks (e.g., "mine wood → craft planks")

### Phase 3: Low-Level Enhancement
1. Add gradient-based MPC to existing LeWorldModel
2. Implement warm-starting with value/policy network
3. Integrate high-level and low-level planners
4. Test on 5-10 step tasks

### Phase 4: Memory and Curriculum
1. Add LSTM/Transformer-XL to encoder
2. Implement curriculum training (progressive horizon)
3. Add shaped costs for planning
4. Test on 10-20 step tasks

### Phase 5: Exploration and Safety
1. Implement curiosity-based data collection
2. Add safety constraints and uncertainty estimation
3. Fine-tune with safety-critic signals
4. Test on full "craft diamond sword" task

## Open Questions
- How to detect subgoals automatically vs. manual specification?
- Training schedule: joint vs. sequential (high-level then low-level)?
- How to evaluate hierarchical planning quality?
- Computational cost: high-level + low-level planning at runtime?
- How to handle failed subgoals (retry, skip, replan)?
- What is the optimal abstraction level for high-level model?

## References
- HWM (Hierarchical World Models)
- THICK (Temporal Hierarchical Context Keypoints)
- C-RSSM (Context-aware Recurrent State-Space Model)
- Dreamer (value/policy networks for warm-starting)
- GTrXL (Gated Transformer-XL for memory)
- WorldRFT (safety-critic for planning)
