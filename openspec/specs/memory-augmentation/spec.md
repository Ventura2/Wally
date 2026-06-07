# Memory Augmentation

## Purpose

Augment the ViT encoder with recurrent memory (LSTM) to produce context-aware latents that incorporate information from recent observation history, enabling the planner to reason over temporal sequences.

## Requirements

### Requirement: Recurrent encoder module
The system SHALL provide a recurrent encoder module that wraps the existing ViT encoder with an LSTM layer to produce context-augmented latents from a sequence of recent observations.

#### Scenario: Encode frame with history context
- **WHEN** a new frame is observed and LSTM hidden state is available
- **THEN** the system encodes the frame with ViT, mean-pools patch tokens, passes the result through LSTM with the current hidden state, and returns a context-augmented latent

#### Scenario: Process frame sequence
- **WHEN** a sequence of T frames is provided
- **THEN** the system processes them sequentially through the LSTM, returning T context-augmented latents and the final hidden state

### Requirement: Hidden state management
The system SHALL maintain and expose LSTM hidden state (h, c) across planning steps, supporting state persistence, retrieval, and reset.

#### Scenario: Persist hidden state across calls
- **WHEN** the recurrent encoder processes a frame and returns a latent
- **THEN** the system updates the internal hidden state and uses it for the next call

#### Scenario: Reset hidden state
- **WHEN** a task completes, the agent dies, or an explicit reset is requested
- **THEN** the system resets the LSTM hidden state to zeros

#### Scenario: Retrieve hidden state
- **WHEN** the current hidden state is requested
- **THEN** the system returns the current (h, c) tuple without modifying it

### Requirement: Configurable memory length
The system SHALL support configurable memory length controlling how many recent frames influence the latent representation, with a default of 16 frames.

#### Scenario: Custom memory length
- **WHEN** a memory length of N is configured
- **THEN** the LSTM processes sequences of up to N frames, with older frames only influencing the latent through the accumulated hidden state

#### Scenario: Single frame mode
- **WHEN** memory length is set to 1
- **THEN** the system behaves like the original ViT encoder (LSTM processes a single frame, hidden state still accumulates)

### Requirement: Backward compatibility with non-recurrent encoder
The system SHALL provide a drop-in replacement interface so that the recurrent encoder can be used anywhere the existing ViT encoder is used, with the LSTM component being optional.

#### Scenario: Use as drop-in replacement
- **WHEN** the recurrent encoder is substituted for ViTEncoder in the planner
- **THEN** the planner functions correctly, with the recurrent encoder's forward method returning latents in the same shape

#### Scenario: Disable recurrence
- **WHEN** recurrence is disabled via configuration
- **THEN** the recurrent encoder returns the same output as the base ViT encoder (bypassing LSTM)
