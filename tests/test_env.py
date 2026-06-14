from unittest.mock import patch

import numpy as np
import pytest
from src.collector.config import CollectorConfig


@pytest.fixture
def mock_sim():
    with patch("src.collector.env._MinecraftSim") as mock:
        sim = mock.return_value
        sim.reset.return_value = (
            {"image": np.zeros((224, 224, 3), dtype=np.uint8)},
            {},
        )
        sim.step.return_value = (
            {"image": np.ones((224, 224, 3), dtype=np.uint8)},
            1.0,
            False,
            False,
            {},
        )
        from src.collector.env import MineStudioEnv

        config = CollectorConfig()
        env = MineStudioEnv(config)
        yield env, sim, mock


class TestMineStudioEnv:
    def test_reset_returns_image_array(self, mock_sim):
        env, sim, _ = mock_sim
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (224, 224, 3)
        sim.reset.assert_called_once()

    def test_step_returns_4_values(self, mock_sim):
        env, sim, _ = mock_sim
        obs, reward, done, info = env.step({"forward": 1})
        assert isinstance(obs, np.ndarray)
        assert reward == 1.0
        assert done is False
        assert info == {}
        sim.step.assert_called_once_with({"forward": 1})

    @pytest.mark.smoke
    def test_step_exposes_pov_in_info(self, mock_sim):
        env, sim, _ = mock_sim
        pov = np.zeros((360, 640, 3), dtype=np.uint8)
        sim.step.return_value = (
            {
                "image": np.ones((224, 224, 3), dtype=np.uint8),
                "pov": pov,
            },
            0.0,
            False,
            False,
            {},
        )
        _, _, _, info = env.step({"forward": 1})
        assert "pov" in info
        assert info["pov"] is pov
        assert info["pov"].shape == (360, 640, 3)

    def test_step_combines_terminated_truncated(self, mock_sim):
        env, sim, _ = mock_sim
        sim.step.return_value = (
            {"image": np.zeros((224, 224, 3), dtype=np.uint8)},
            0.0,
            True,
            False,
            {},
        )
        _, _, done, _ = env.step({})
        assert done is True

        sim.step.return_value = (
            {"image": np.zeros((224, 224, 3), dtype=np.uint8)},
            0.0,
            False,
            True,
            {},
        )
        _, _, done, _ = env.step({})
        assert done is True

    def test_action_space_property(self, mock_sim):
        env, sim, _ = mock_sim
        sim.action_space = {"forward": 1, "jump": 0}
        assert env.action_space == {"forward": 1, "jump": 0}

    def test_close(self, mock_sim):
        env, sim, _ = mock_sim
        env.close()
        sim.close.assert_called_once()

    def test_raises_when_minestudio_missing(self):
        with patch("src.collector.env._MinecraftSim", None):
            with pytest.raises(ImportError, match="MineStudio is not installed"):
                from src.collector.env import MineStudioEnv

                MineStudioEnv(CollectorConfig())
