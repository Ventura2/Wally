# Smoke tests

Smoke tests in this project are fast pytest tests marked with `@pytest.mark.smoke` and run as a pre-commit gate on Windows. They cover the training pipeline, hierarchy, planner, agent loop, deployer, and docs freshness.

## How it works

```mermaid
flowchart TB
    subgraph Define["pyproject.toml:48"]
        M["pytest marker:<br/>smoke: fast smoke tests for CI"]
    end

    subgraph Run["Run"]
        CMD["pytest -m smoke -x --tb=short<br/>(Windows venv, before every commit)"]
    end

    subgraph Conftest["tests/conftest.py"]
        PATH["adds src/ to sys.path"]
    end

    subgraph Tests["tests/test_*.py (42 pass, 8 skip)"]
        T1["test_agent_loop.py"]
        T2["test_deployer_cli.py"]
        T3["test_deployer_mock_env.py"]
        T4["test_env.py"]
        T5["test_hierarchy_smoke.py"]
        T6["test_lewm_numerical_stability.py"]
        T7["test_lewm_residual_loss.py"]
        T8["test_planner_smoke.py"]
        T9["test_play_cli.py"]
        T10["test_server_env.py"]
        T11["test_train_logging.py"]
        T12["test_training_utils.py"]
        T13["test_viewer.py"]
        T14["test_wood_gathering_regression.py"]
        T15["test_docs_freshness.py"]
    end

    subgraph Areas["Subsystems covered"]
        A1[Training pipeline<br/>logging, numerics, residuals]
        A2[Hierarchy L0-L3 wiring]
        A3[CEM planner end-to-end]
        A4[Agent loop + viewer]
        A5[Deployer CLI / server env]
        A6[Docs freshness]
    end

    M --> CMD
    PATH --> Tests
    CMD --> Tests
    Tests --> Areas
```

## Key points

- **Marker**: `pytest.mark.smoke` declared in `pyproject.toml:48` (`markers = ["smoke: fast smoke tests for CI"]`); opt-in decorator on individual test functions.
- **Command**: `.\.venv-windows\Scripts\python.exe -m pytest -m smoke -x --tb=short` (see `AGENTS.md:143`). Run before every commit.
- **Path setup**: `tests/conftest.py` prepends `src/` to `sys.path` so `from wally...` imports resolve without an editable install.
- **Scope**: ~42 tests across training stability, hierarchy wiring, planner, agent loop, deployer, viewer, and docs freshness.
- **CPU escape hatch**: a handful of fast smoke tests run on CPU via `--device cpu`; production training never falls back to CPU (`src/wally/AGENTS.md`).

## Related: `wally-plan-smoke` CLI

Distinct from the pytest suite. A one-shot world-model sanity check that loads a checkpoint and runs the CEM planner on two synthetic frames:

```bash
uv run wally-plan-smoke
uv run wally-plan-smoke --checkpoint checkpoints/checkpoint_500.pt --output /tmp/probe.pt
```

Defined in `pyproject.toml:39` and implemented in `src/wally/planner/plan_smoke_cli.py`.
