from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wally.planner.config import CEMConfig


class TestCEMConfigDefaults:
    def test_default_values(self):
        cfg = CEMConfig.default()
        assert cfg.population_size == 64
        assert cfg.elite_frac == 0.1
        assert cfg.n_iterations == 5
        assert cfg.horizon == 8
        assert cfg.action_low == -1.0
        assert cfg.action_high == 1.0
        assert cfg.gradient_policy == "detach"

    def test_constructor_defaults_match_documented(self):
        cfg = CEMConfig()
        assert cfg.population_size == 64
        assert cfg.elite_frac == 0.1
        assert cfg.n_iterations == 5
        assert cfg.horizon == 8
        assert cfg.action_low == -1.0
        assert cfg.action_high == 1.0


class TestCEMConfigValidation:
    def test_elite_frac_zero_fails(self):
        with pytest.raises(ValidationError, match="elite_frac"):
            CEMConfig(elite_frac=0.0)

    def test_elite_frac_one_fails(self):
        with pytest.raises(ValidationError, match="elite_frac"):
            CEMConfig(elite_frac=1.0)

    def test_elite_frac_negative_fails(self):
        with pytest.raises(ValidationError, match="elite_frac"):
            CEMConfig(elite_frac=-0.1)

    def test_population_size_one_fails(self):
        with pytest.raises(ValidationError, match="population_size"):
            CEMConfig(population_size=1)

    def test_population_size_zero_fails(self):
        with pytest.raises(ValidationError, match="population_size"):
            CEMConfig(population_size=0)

    def test_n_iterations_zero_fails(self):
        with pytest.raises(ValidationError, match="n_iterations"):
            CEMConfig(n_iterations=0)

    def test_horizon_zero_fails(self):
        with pytest.raises(ValidationError, match="horizon"):
            CEMConfig(horizon=0)

    def test_invalid_gradient_policy_fails(self):
        with pytest.raises(ValidationError):
            CEMConfig(gradient_policy="invalid")

    def test_valid_gradient_policies(self):
        assert CEMConfig(gradient_policy="detach").gradient_policy == "detach"
        cfg = CEMConfig(gradient_policy="straight_through")
        assert cfg.gradient_policy == "straight_through"


class TestCEMConfigFromYaml:
    def test_load_valid_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "cem.yaml"
        yaml_file.write_text(
            "population_size: 128\n"
            "elite_frac: 0.2\n"
            "n_iterations: 10\n"
            "horizon: 16\n"
            "action_low: -2.0\n"
            "action_high: 2.0\n"
            "gradient_policy: straight_through\n"
        )
        cfg = CEMConfig.from_yaml(yaml_file)
        assert cfg.population_size == 128
        assert cfg.elite_frac == 0.2
        assert cfg.n_iterations == 10
        assert cfg.horizon == 16
        assert cfg.action_low == -2.0
        assert cfg.action_high == 2.0
        assert cfg.gradient_policy == "straight_through"

    def test_missing_fields_use_defaults(self, tmp_path: Path):
        yaml_file = tmp_path / "cem.yaml"
        yaml_file.write_text("population_size: 32\n")
        cfg = CEMConfig.from_yaml(yaml_file)
        assert cfg.population_size == 32
        assert cfg.elite_frac == 0.1
        assert cfg.n_iterations == 5
        assert cfg.horizon == 8

    def test_empty_yaml_uses_all_defaults(self, tmp_path: Path):
        yaml_file = tmp_path / "cem.yaml"
        yaml_file.write_text("")
        cfg = CEMConfig.from_yaml(yaml_file)
        assert cfg == CEMConfig.default()

    def test_invalid_yaml_value_fails(self, tmp_path: Path):
        yaml_file = tmp_path / "cem.yaml"
        yaml_file.write_text("elite_frac: 0.0\n")
        with pytest.raises(ValidationError, match="elite_frac"):
            CEMConfig.from_yaml(yaml_file)
