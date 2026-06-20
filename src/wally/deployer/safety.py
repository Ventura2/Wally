"""Safety filters for action validation."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from wally.deployer.config import SafetyConfig

logger = logging.getLogger(__name__)

BEDROCK_BLOCK_ID = 7
LAVA_BLOCK_IDS = {10, 11}


@dataclass
class ActionContext:
    action_type: str
    target_block_id: int | None = None
    target_position: tuple[int, int, int] | None = None
    adjacent_block_ids: list[int] | None = None
    player_position: tuple[float, float, float] | None = None


class SafetyFilterBase(ABC):
    @abstractmethod
    def check(self, ctx: ActionContext) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class BedrockFilter(SafetyFilterBase):
    @property
    def name(self) -> str:
        return "bedrock"

    def check(self, ctx: ActionContext) -> bool:
        if ctx.action_type in ("break",) and ctx.target_block_id == BEDROCK_BLOCK_ID:
            logger.warning(
                "BedrockFilter: blocked breaking bedrock at %s",
                ctx.target_position,
            )
            return False
        return True


class LavaFilter(SafetyFilterBase):
    @property
    def name(self) -> str:
        return "lava"

    def check(self, ctx: ActionContext) -> bool:
        if ctx.action_type == "place" and ctx.adjacent_block_ids:
            if any(bid in LAVA_BLOCK_IDS for bid in ctx.adjacent_block_ids):
                logger.warning(
                    "LavaFilter: blocked placement adjacent to lava at %s",
                    ctx.target_position,
                )
                return False
        return True


class VoidFilter(SafetyFilterBase):
    def __init__(self, threshold: float = -64.0) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "void"

    def check(self, ctx: ActionContext) -> bool:
        if ctx.player_position and ctx.player_position[1] < self._threshold:
            logger.warning(
                "VoidFilter: player below void threshold y=%.1f < %.1f",
                ctx.player_position[1],
                self._threshold,
            )
            return False
        return True


class CooldownFilter(SafetyFilterBase):
    def __init__(self, cooldown_ms: int = 100) -> None:
        self._cooldown_s = cooldown_ms / 1000.0
        self._last_action_time: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "cooldown"

    def check(self, ctx: ActionContext) -> bool:
        now = time.monotonic()
        last = self._last_action_time.get(ctx.action_type, 0.0)
        if now - last < self._cooldown_s:
            logger.warning(
                "CooldownFilter: action '%s' within cooldown window",
                ctx.action_type,
            )
            return False
        self._last_action_time[ctx.action_type] = now
        return True


class SafetyFilter:
    def __init__(self, config: SafetyConfig | None = None) -> None:
        self._filters: dict[str, SafetyFilterBase] = {}
        self._enabled: dict[str, bool] = {}
        self._violations: list[str] = []
        cfg = config or SafetyConfig()
        self.register(BedrockFilter(), cfg.prevent_bedrock_breaking)
        self.register(LavaFilter(), cfg.prevent_lava_interaction)
        self.register(VoidFilter(threshold=cfg.void_threshold), cfg.prevent_void_fall)
        self.register(CooldownFilter(cooldown_ms=cfg.action_cooldown_ms), True)

    def register(self, filter_: SafetyFilterBase, enabled: bool = True) -> None:
        self._filters[filter_.name] = filter_
        self._enabled[filter_.name] = enabled

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name in self._enabled:
            self._enabled[name] = enabled

    def check(self, ctx: ActionContext) -> bool:
        for name, filter_ in self._filters.items():
            if self._enabled.get(name, True) and not filter_.check(ctx):
                self._violations.append(f"{name}: blocked {ctx.action_type}")
                return False
        return True

    def get_violation_log(self) -> list[str]:
        return list(self._violations)
