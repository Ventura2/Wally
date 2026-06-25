"""Hierarchical JEPA world models (L1, L2, L3) on top of the L0 LeWorldModel.

This package implements a stack of learned world models, one per
abstraction level, that predict the next state at that level's time
horizon and communicate by streaming continuous ``Tensor[D]`` embeddings.
The L0 LeWorldModel is treated as a frozen black box at the bottom of the
stack; L1+ are pure JEPA predictors in their own embedding spaces.
"""

from wally.hierarchy.config import HierarchyConfig, LayerSpec
from wally.hierarchy.encoders import L1Encoder, L2Encoder, L3Encoder
from wally.hierarchy.goal import LearnedGoalEmbedding
from wally.hierarchy.jepa import JEPAWorldModel
from wally.hierarchy.types import LayerMessage, LayerState

__all__ = [
    "HierarchyConfig",
    "JEPAWorldModel",
    "L1Encoder",
    "L2Encoder",
    "L3Encoder",
    "LayerMessage",
    "LayerSpec",
    "LayerState",
    "LearnedGoalEmbedding",
]
