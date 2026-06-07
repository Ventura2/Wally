## ADDED Requirements

### Requirement: Replanning strategy
The MineStudio integration SHALL support a configurable replanning strategy that determines when the planner is re-invoked during an episode.

#### Scenario: Fixed-interval replanning
- **WHEN** the agent has executed `replan_interval` steps since the last plan
- **THEN** the planner SHALL be re-invoked with the current observed frame and the original goal frame

#### Scenario: Replanning at episode start
- **WHEN** an episode begins
- **THEN** the planner SHALL be invoked immediately before any actions are executed

### Requirement: Action execution
The MineStudio integration SHALL translate planner output into MineStudio-compatible actions and execute them in the environment.

#### Scenario: Continuous to discrete action translation
- **WHEN** the planner produces a continuous action vector of shape `(25,)`
- **THEN** the system SHALL convert it to a discrete action dict using `MineStudioActionVocab` and `continuous_to_discrete()`

#### Scenario: Sequential action execution
- **WHEN** a plan contains H actions
- **THEN** the system SHALL execute actions one at a time via `env.step()`, observing the resulting frame after each step

### Requirement: Episode termination
The MineStudio integration SHALL define clear episode termination criteria.

#### Scenario: Environment signals done
- **WHEN** `env.step()` returns `done=True`
- **THEN** the episode SHALL terminate immediately

#### Scenario: Step timeout
- **WHEN** the total number of executed steps reaches `episode_timeout`
- **THEN** the episode SHALL terminate with a timeout indicator

### Requirement: Goal specification
The MineStudio integration SHALL accept goal specifications as RGB frames.

#### Scenario: Frame goal input
- **WHEN** an episode is started with a goal frame
- **THEN** the goal frame SHALL be a numpy array or tensor of shape `(H, W, 3)` or `(3, H, W)` with uint8 or float pixel values
- **THEN** the goal frame SHALL be encoded to a latent representation once and reused for all planning calls during the episode
