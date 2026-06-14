## 1. Wire deterministic run name into `init_wandb`

- [x] 1.1 Add `name: str | None = None` parameter to `init_wandb` in `src/wally/training/logging.py` and forward it to `wandb.init(...)`.

## 2. Pass name from trainer call site

- [x] 2.1 In `Trainer.train()` (`src/wally/training/trainer.py:219`), compute `run_name = f"{self.config['wandb_project']}-step-{self.global_step}"` and pass it as `name=run_name` to `init_wandb`.

## 3. Test

- [x] 3.1 Add a smoke test in `tests/test_train_logging.py` that patches `wally.training.logging.wandb.init` and asserts the `name=` kwarg matches the expected `f"{wandb_project}-step-{global_step}"` pattern for both `global_step=0` and a resumed `global_step=50000`.

## 4. Verify

- [x] 4.1 Run smoke tests: `.\.venv-windows\Scripts\python.exe -m pytest -m smoke -x --tb=short`
- [x] 4.2 Run lint: `.\.venv-windows\Scripts\python.exe -m ruff check .`
- [x] 4.3 Run typecheck: `.\.venv-windows\Scripts\python.exe -m mypy` (pre-existing error in `src/agent/loop.py` — src-layout module-naming conflict, not introduced by this change; touched files are type-clean)
