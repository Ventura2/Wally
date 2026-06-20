from __future__ import annotations

from unittest.mock import patch

from wally.deployer.config import SafetyConfig


class TestBedrockFilter:
    def test_blocks_breaking_bedrock(self):
        from wally.deployer.safety import ActionContext, BedrockFilter

        f = BedrockFilter()
        ctx = ActionContext(
            action_type="break",
            target_block_id=7,
            target_position=(0, 0, 0),
        )
        assert f.check(ctx) is False

    def test_allows_breaking_non_bedrock(self):
        from wally.deployer.safety import ActionContext, BedrockFilter

        f = BedrockFilter()
        ctx = ActionContext(
            action_type="break",
            target_block_id=1,
            target_position=(0, 0, 0),
        )
        assert f.check(ctx) is True

    def test_allows_non_break_actions_on_bedrock(self):
        from wally.deployer.safety import ActionContext, BedrockFilter

        f = BedrockFilter()
        ctx = ActionContext(
            action_type="place",
            target_block_id=7,
            target_position=(0, 0, 0),
        )
        assert f.check(ctx) is True

    def test_name_is_bedrock(self):
        from wally.deployer.safety import BedrockFilter

        assert BedrockFilter().name == "bedrock"


class TestLavaFilter:
    def test_blocks_placement_adjacent_to_lava(self):
        from wally.deployer.safety import ActionContext, LavaFilter

        f = LavaFilter()
        ctx = ActionContext(
            action_type="place",
            adjacent_block_ids=[1, 10, 3],
            target_position=(1, 2, 3),
        )
        assert f.check(ctx) is False

    def test_blocks_placement_adjacent_to_still_lava(self):
        from wally.deployer.safety import ActionContext, LavaFilter

        f = LavaFilter()
        ctx = ActionContext(
            action_type="place",
            adjacent_block_ids=[1, 2, 11],
            target_position=(1, 2, 3),
        )
        assert f.check(ctx) is False

    def test_allows_placement_not_adjacent_to_lava(self):
        from wally.deployer.safety import ActionContext, LavaFilter

        f = LavaFilter()
        ctx = ActionContext(
            action_type="place",
            adjacent_block_ids=[1, 2, 3],
            target_position=(1, 2, 3),
        )
        assert f.check(ctx) is True

    def test_allows_non_place_actions_near_lava(self):
        from wally.deployer.safety import ActionContext, LavaFilter

        f = LavaFilter()
        ctx = ActionContext(
            action_type="break",
            adjacent_block_ids=[10, 11],
            target_position=(1, 2, 3),
        )
        assert f.check(ctx) is True

    def test_allows_place_with_no_adjacent_info(self):
        from wally.deployer.safety import ActionContext, LavaFilter

        f = LavaFilter()
        ctx = ActionContext(action_type="place", adjacent_block_ids=None)
        assert f.check(ctx) is True

    def test_name_is_lava(self):
        from wally.deployer.safety import LavaFilter

        assert LavaFilter().name == "lava"


class TestVoidFilter:
    def test_blocks_when_player_below_threshold(self):
        from wally.deployer.safety import ActionContext, VoidFilter

        f = VoidFilter(threshold=-64.0)
        ctx = ActionContext(
            action_type="move",
            player_position=(0.0, -65.0, 0.0),
        )
        assert f.check(ctx) is False

    def test_allows_when_player_above_threshold(self):
        from wally.deployer.safety import ActionContext, VoidFilter

        f = VoidFilter(threshold=-64.0)
        ctx = ActionContext(
            action_type="move",
            player_position=(0.0, 64.0, 0.0),
        )
        assert f.check(ctx) is True

    def test_allows_when_player_at_threshold(self):
        from wally.deployer.safety import ActionContext, VoidFilter

        f = VoidFilter(threshold=-64.0)
        ctx = ActionContext(
            action_type="move",
            player_position=(0.0, -64.0, 0.0),
        )
        assert f.check(ctx) is True

    def test_custom_threshold(self):
        from wally.deployer.safety import ActionContext, VoidFilter

        f = VoidFilter(threshold=0.0)
        ctx_below = ActionContext(
            action_type="move",
            player_position=(0.0, -1.0, 0.0),
        )
        ctx_above = ActionContext(
            action_type="move",
            player_position=(0.0, 1.0, 0.0),
        )
        assert f.check(ctx_below) is False
        assert f.check(ctx_above) is True

    def test_allows_when_no_player_position(self):
        from wally.deployer.safety import ActionContext, VoidFilter

        f = VoidFilter()
        ctx = ActionContext(action_type="move", player_position=None)
        assert f.check(ctx) is True

    def test_name_is_void(self):
        from wally.deployer.safety import VoidFilter

        assert VoidFilter().name == "void"


class TestCooldownFilter:
    def test_blocks_action_within_cooldown_window(self):
        from wally.deployer.safety import ActionContext, CooldownFilter

        f = CooldownFilter(cooldown_ms=100)
        ctx = ActionContext(action_type="break")
        assert f.check(ctx) is True
        assert f.check(ctx) is False

    def test_allows_action_after_cooldown_expires(self):
        from wally.deployer.safety import ActionContext, CooldownFilter

        f = CooldownFilter(cooldown_ms=50)
        ctx = ActionContext(action_type="break")

        with patch("wally.deployer.safety.time") as mock_time:
            mock_time.monotonic.return_value = 1.0
            assert f.check(ctx) is True

            mock_time.monotonic.return_value = 1.1
            assert f.check(ctx) is True

    def test_different_action_types_have_independent_cooldowns(self):
        from wally.deployer.safety import ActionContext, CooldownFilter

        f = CooldownFilter(cooldown_ms=100)
        ctx_break = ActionContext(action_type="break")
        ctx_place = ActionContext(action_type="place")

        assert f.check(ctx_break) is True
        assert f.check(ctx_place) is True
        assert f.check(ctx_break) is False
        assert f.check(ctx_place) is False

    def test_name_is_cooldown(self):
        from wally.deployer.safety import CooldownFilter

        assert CooldownFilter().name == "cooldown"


class TestSafetyFilterComposite:
    def test_all_filters_enabled_by_default(self):
        sf = __import__("wally.deployer.safety", fromlist=["SafetyFilter"]).SafetyFilter()
        ctx = __import__("wally.deployer.safety", fromlist=["ActionContext"]).ActionContext(
            action_type="break",
            target_block_id=7,
            target_position=(0, 0, 0),
        )
        assert sf.check(ctx) is False

    def test_disabling_filter_allows_previously_blocked_action(self):
        from wally.deployer.safety import ActionContext, SafetyFilter

        sf = SafetyFilter()
        sf.set_enabled("bedrock", False)
        ctx = ActionContext(
            action_type="break",
            target_block_id=7,
            target_position=(0, 0, 0),
        )
        assert sf.check(ctx) is True

    def test_violation_logging(self):
        from wally.deployer.safety import ActionContext, SafetyFilter

        sf = SafetyFilter()
        ctx = ActionContext(
            action_type="break",
            target_block_id=7,
            target_position=(0, 0, 0),
        )
        sf.check(ctx)
        violations = sf.get_violation_log()
        assert len(violations) == 1
        assert "bedrock" in violations[0]
        assert "break" in violations[0]

    def test_config_driven_disable_bedrock(self):
        from wally.deployer.safety import ActionContext, SafetyFilter

        cfg = SafetyConfig(prevent_bedrock_breaking=False)
        sf = SafetyFilter(config=cfg)
        ctx = ActionContext(
            action_type="break",
            target_block_id=7,
            target_position=(0, 0, 0),
        )
        assert sf.check(ctx) is True

    def test_config_driven_disable_lava(self):
        from wally.deployer.safety import ActionContext, SafetyFilter

        cfg = SafetyConfig(prevent_lava_interaction=False)
        sf = SafetyFilter(config=cfg)
        ctx = ActionContext(
            action_type="place",
            adjacent_block_ids=[10],
            target_position=(1, 2, 3),
        )
        assert sf.check(ctx) is True

    def test_config_driven_disable_void(self):
        from wally.deployer.safety import ActionContext, SafetyFilter

        cfg = SafetyConfig(prevent_void_fall=False)
        sf = SafetyFilter(config=cfg)
        ctx = ActionContext(
            action_type="move",
            player_position=(0.0, -100.0, 0.0),
        )
        assert sf.check(ctx) is True

    def test_safe_action_passes_all_filters(self):
        from wally.deployer.safety import ActionContext, SafetyFilter

        sf = SafetyFilter()
        ctx = ActionContext(
            action_type="move",
            player_position=(0.0, 64.0, 0.0),
        )
        assert sf.check(ctx) is True

    def test_multiple_violations_tracked(self):
        from wally.deployer.safety import ActionContext, SafetyFilter

        sf = SafetyFilter()
        ctx1 = ActionContext(
            action_type="break",
            target_block_id=7,
            target_position=(0, 0, 0),
        )
        ctx2 = ActionContext(
            action_type="place",
            adjacent_block_ids=[10],
            target_position=(1, 2, 3),
        )
        sf.check(ctx1)
        sf.check(ctx2)
        violations = sf.get_violation_log()
        assert len(violations) == 2

    def test_register_custom_filter(self):
        from wally.deployer.safety import (
            ActionContext,
            SafetyFilter,
            SafetyFilterBase,
        )

        class AlwaysBlockFilter(SafetyFilterBase):
            @property
            def name(self) -> str:
                return "always_block"

            def check(self, ctx: ActionContext) -> bool:
                return False

        sf = SafetyFilter()
        sf.register(AlwaysBlockFilter(), enabled=True)
        ctx = ActionContext(action_type="move")
        assert sf.check(ctx) is False

    def test_violation_log_returns_copy(self):
        from wally.deployer.safety import SafetyFilter

        sf = SafetyFilter()
        log1 = sf.get_violation_log()
        log1.append("fake")
        assert len(sf.get_violation_log()) == 0
