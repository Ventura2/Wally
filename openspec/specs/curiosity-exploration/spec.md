# Curiosity Exploration

## Purpose

Provide intrinsic curiosity-driven exploration for the world model training pipeline, using prediction error of a forward dynamics model as an intrinsic reward signal to prioritize novel states.

## Requirements

### Requirement: Intrinsic curiosity reward computation
The system SHALL compute intrinsic curiosity rewards as the prediction error of a forward model that predicts the next-state latent from the current-state latent and action.

#### Scenario: Compute curiosity reward for a transition
- **WHEN** a (current_latent, action, next_latent) tuple is provided
- **THEN** the system returns the L2 distance between the predicted next latent and the actual next latent as the intrinsic reward

#### Scenario: Batched curiosity rewards
- **WHEN** a batch of transitions is provided
- **THEN** the system returns per-sample intrinsic rewards

### Requirement: Forward model training
The system SHALL train a forward dynamics model (small MLP or Transformer head) that maps (current_latent, action) -> predicted_next_latent, using the same latent space as the world model.

#### Scenario: Train forward model on collected data
- **WHEN** trajectory data with encoded latents is available
- **THEN** the system trains the forward model to minimize prediction error on (z_t, a_t) -> z_{t+1}

#### Scenario: Shared latent space
- **WHEN** the forward model is trained
- **THEN** the system uses latents from the existing ViT encoder (frozen or fine-tuned), not a separate representation

### Requirement: Exploration policy integration
The system SHALL support integrating intrinsic rewards with the data collection process to prioritize collection of trajectories in high-curiosity (novel) regions of the state space.

#### Scenario: Curiosity-weighted data collection
- **WHEN** collecting new training trajectories
- **THEN** the system assigns higher collection priority to environment states with high prediction error under the current forward model

#### Scenario: Decay curiosity over training
- **WHEN** the forward model improves and prediction errors decrease
- **THEN** the system naturally shifts exploration toward remaining novel states as previously explored states become predictable

### Requirement: Curiosity module configuration
The system SHALL support configurable curiosity parameters including forward model architecture, reward scaling factor, and update frequency relative to world model training.

#### Scenario: Custom reward scaling
- **WHEN** a reward scaling factor is configured
- **THEN** the system multiplies raw prediction errors by that factor before using them as intrinsic rewards

#### Scenario: Configuration validation
- **WHEN** invalid parameters are provided (e.g., negative scaling)
- **THEN** the system raises a validation error
