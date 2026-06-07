from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from wally.cli.train import parse_args


class TestParseArgs:
    def test_config_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_config_path(self):
        args = parse_args(["--config", "configs/lewm_default.yaml"])
        assert args.config == Path("configs/lewm_default.yaml")

    def test_default_device(self):
        args = parse_args(["--config", "test.yaml"])
        assert args.device == "auto"

    def test_device_cpu(self):
        args = parse_args(["--config", "test.yaml", "--device", "cpu"])
        assert args.device == "cpu"

    def test_device_cuda(self):
        args = parse_args(["--config", "test.yaml", "--device", "cuda"])
        assert args.device == "cuda"

    def test_resume_path(self):
        args = parse_args(["--config", "test.yaml", "--resume", "ckpt.pt"])
        assert args.resume == Path("ckpt.pt")

    def test_resume_default_none(self):
        args = parse_args(["--config", "test.yaml"])
        assert args.resume is None

    def test_invalid_device(self):
        with pytest.raises(SystemExit):
            parse_args(["--config", "test.yaml", "--device", "tpu"])
