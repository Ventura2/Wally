from __future__ import annotations

import logging

import numpy as np

from wally.deployer.executor import ACTION_DIM, ActionExecutor


class TestActionValidation:
    def test_valid_action_passes(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        assert executor.validate(action) is True

    def test_valid_action_with_values_passes(self):
        executor = ActionExecutor()
        action = np.ones(ACTION_DIM) * 0.5
        assert executor.validate(action) is True

    def test_wrong_shape_rejected(self):
        executor = ActionExecutor()
        action = np.zeros(10)
        assert executor.validate(action) is False

    def test_wrong_shape_2d_rejected(self):
        executor = ActionExecutor()
        action = np.zeros((1, ACTION_DIM))
        assert executor.validate(action) is False

    def test_out_of_bounds_rejected(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[0] = 2.0
        assert executor.validate(action) is False

    def test_negative_out_of_bounds_rejected(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[5] = -1.5
        assert executor.validate(action) is False

    def test_boundary_value_accepted(self):
        executor = ActionExecutor()
        action = np.ones(ACTION_DIM)
        assert executor.validate(action) is True

    def test_warning_logged_on_invalid_shape(self, caplog):
        executor = ActionExecutor()
        action = np.zeros(10)
        with caplog.at_level(logging.WARNING, logger="deployer.executor"):
            executor.validate(action)
        assert any("Invalid action shape" in r.message for r in caplog.records)

    def test_warning_logged_on_out_of_bounds(self, caplog):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[0] = 5.0
        with caplog.at_level(logging.WARNING, logger="deployer.executor"):
            executor.validate(action)
        assert any("out of bounds" in r.message for r in caplog.records)


class TestMovementTranslation:
    def test_forward_produces_movement_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[0] = 1.0
        packets = executor._translate_movement(action)
        movement = [p for p in packets if p["type"] == "movement"]
        assert len(movement) == 1
        assert movement[0]["dx"] == 1.0
        assert movement[0]["dz"] == 0.0

    def test_backward_produces_movement_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[1] = 1.0
        packets = executor._translate_movement(action)
        movement = [p for p in packets if p["type"] == "movement"]
        assert len(movement) == 1
        assert movement[0]["dx"] == -1.0

    def test_strafe_right_produces_movement_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[3] = 1.0
        packets = executor._translate_movement(action)
        movement = [p for p in packets if p["type"] == "movement"]
        assert len(movement) == 1
        assert movement[0]["dz"] == 1.0

    def test_strafe_left_produces_movement_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[2] = 1.0
        packets = executor._translate_movement(action)
        movement = [p for p in packets if p["type"] == "movement"]
        assert len(movement) == 1
        assert movement[0]["dz"] == -1.0

    def test_turn_right_produces_rotation_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[8] = 1.0
        packets = executor._translate_movement(action)
        rotation = [p for p in packets if p["type"] == "rotation"]
        assert len(rotation) == 1
        assert rotation[0]["dyaw"] == 10.0

    def test_turn_left_produces_rotation_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[7] = 1.0
        packets = executor._translate_movement(action)
        rotation = [p for p in packets if p["type"] == "rotation"]
        assert len(rotation) == 1
        assert rotation[0]["dyaw"] == -10.0

    def test_pitch_down_produces_rotation_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[10] = 1.0
        packets = executor._translate_movement(action)
        rotation = [p for p in packets if p["type"] == "rotation"]
        assert len(rotation) == 1
        assert rotation[0]["dpitch"] == 10.0

    def test_pitch_up_produces_rotation_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[9] = 1.0
        packets = executor._translate_movement(action)
        rotation = [p for p in packets if p["type"] == "rotation"]
        assert len(rotation) == 1
        assert rotation[0]["dpitch"] == -10.0

    def test_jump_produces_jump_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[4] = 1.0
        packets = executor._translate_movement(action)
        assert any(p["type"] == "jump" for p in packets)

    def test_sneak_produces_sneak_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[5] = 1.0
        packets = executor._translate_movement(action)
        assert any(p["type"] == "sneak" for p in packets)

    def test_sprint_produces_sprint_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[6] = 1.0
        packets = executor._translate_movement(action)
        assert any(p["type"] == "sprint" for p in packets)

    def test_zero_action_produces_no_packets(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        packets = executor._translate_movement(action)
        assert len(packets) == 0

    def test_below_threshold_produces_no_packets(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[0] = 0.3
        packets = executor._translate_movement(action)
        assert len(packets) == 0


class TestBlockInteraction:
    def test_attack_produces_dig_packets(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[11] = 1.0
        packets = executor._translate_block_interaction(action)
        types = [p["type"] for p in packets]
        assert "dig_start" in types
        assert "dig_stop" in types

    def test_use_produces_place_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[12] = 1.0
        packets = executor._translate_block_interaction(action)
        assert any(p["type"] == "place_block" for p in packets)

    def test_pick_produces_pick_block_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[13] = 1.0
        packets = executor._translate_block_interaction(action)
        assert any(p["type"] == "pick_block" for p in packets)

    def test_no_interaction_produces_no_packets(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        packets = executor._translate_block_interaction(action)
        assert len(packets) == 0

    def test_below_threshold_produces_no_packets(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[11] = 0.3
        packets = executor._translate_block_interaction(action)
        assert len(packets) == 0


class TestInventoryTranslation:
    def test_craft_produces_craft_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[14] = 1.0
        packets = executor._translate_inventory(action)
        assert any(p["type"] == "craft" for p in packets)

    def test_slot_selection_produces_select_slot_packet(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[15] = 1.0
        packets = executor._translate_inventory(action)
        slot_packets = [p for p in packets if p["type"] == "select_slot"]
        assert len(slot_packets) == 1
        assert slot_packets[0]["slot"] == 0

    def test_slot_5_selection(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[20] = 1.0
        packets = executor._translate_inventory(action)
        slot_packets = [p for p in packets if p["type"] == "select_slot"]
        assert len(slot_packets) == 1
        assert slot_packets[0]["slot"] == 5

    def test_multiple_slots_selects_first(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[15] = 1.0
        action[17] = 1.0
        packets = executor._translate_inventory(action)
        slot_packets = [p for p in packets if p["type"] == "select_slot"]
        assert len(slot_packets) == 1
        assert slot_packets[0]["slot"] == 0

    def test_no_inventory_action_produces_no_packets(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        packets = executor._translate_inventory(action)
        assert len(packets) == 0


class TestActionExecutorExecute:
    def test_execute_calls_validate_first(self):
        executor = ActionExecutor()
        action = np.zeros(10)
        result = executor.execute(action)
        assert result == []

    def test_execute_returns_empty_for_invalid_action(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[0] = 5.0
        result = executor.execute(action)
        assert result == []

    def test_execute_combines_movement_and_interaction(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[0] = 1.0
        action[11] = 1.0
        result = executor.execute(action)
        types = [p["type"] for p in result]
        assert "movement" in types
        assert "dig_start" in types
        assert "dig_stop" in types

    def test_execute_combines_all_translations(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        action[0] = 1.0
        action[11] = 1.0
        action[14] = 1.0
        action[15] = 1.0
        result = executor.execute(action)
        types = [p["type"] for p in result]
        assert "movement" in types
        assert "dig_start" in types
        assert "craft" in types
        assert "select_slot" in types

    def test_execute_with_valid_zero_action(self):
        executor = ActionExecutor()
        action = np.zeros(ACTION_DIM)
        result = executor.execute(action)
        assert result == []

    def test_connection_property(self):
        executor = ActionExecutor()
        assert executor.connection is None
        executor.connection = "mock_conn"
        assert executor.connection == "mock_conn"

    def test_connection_via_constructor(self):
        executor = ActionExecutor(connection="mock_conn")
        assert executor.connection == "mock_conn"
