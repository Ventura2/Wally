"""Inter-layer message types for the hierarchical world-model stack.

All inter-layer communication is continuous: only ``Tensor[D]`` fields, no
strings, no symbolic task names, no discrete skill IDs. This matches the
``layer-communication-protocol`` spec — runtime messages between layers
contain nothing the system could pre-define as a vocabulary.
"""

from __future__ import annotations

import torch


class LayerState:
    """A layer's current belief about the world.

    Attributes:
        actual_s: The most recent actual state embedding received from the
            layer below. ``None`` until the first message arrives.
        predicted_s: The layer's most recent predicted state embedding, used
            to compute drift against the next ``actual_s``.
        target_embedding: The current target embedding from the layer above.
            ``None`` until a plan has been issued.
        drift: L2 distance between the most recent ``actual_s`` and the
            most recent ``predicted_s``. ``None`` until at least two
            updates have been received.
    """

    __slots__ = ("actual_s", "predicted_s", "target_embedding", "drift")

    def __init__(
        self,
        actual_s: torch.Tensor | None = None,
        predicted_s: torch.Tensor | None = None,
        target_embedding: torch.Tensor | None = None,
        drift: float | None = None,
    ) -> None:
        self.actual_s = actual_s
        self.predicted_s = predicted_s
        self.target_embedding = target_embedding
        self.drift = drift


class LayerMessage:
    """A continuous-embedding message between two layers.

    Top-down messages carry a ``target_embedding``; bottom-up messages
    carry a ``state_embedding`` and an optional ``drift`` scalar.

    Attributes:
        state_embedding: The actual state embedding produced by the sender
            (bottom-up only). ``None`` for top-down messages.
        target_embedding: The target embedding the sender wants the
            receiver to steer toward (top-down only). ``None`` for
            bottom-up messages.
        drift: L2 distance between the sender's predicted and actual state
            embeddings, included on bottom-up messages so the receiver
            can react without a separate query.
    """

    __slots__ = ("state_embedding", "target_embedding", "drift")

    def __init__(
        self,
        state_embedding: torch.Tensor | None = None,
        target_embedding: torch.Tensor | None = None,
        drift: float | None = None,
    ) -> None:
        self.state_embedding = state_embedding
        self.target_embedding = target_embedding
        self.drift = drift

    @classmethod
    def from_target(cls, target: torch.Tensor) -> "LayerMessage":
        return cls(target_embedding=target)

    @classmethod
    def from_state(
        cls, state: torch.Tensor, drift: float | None = None
    ) -> "LayerMessage":
        return cls(state_embedding=state, drift=drift)
