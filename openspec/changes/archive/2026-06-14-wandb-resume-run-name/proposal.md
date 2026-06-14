## Why

When a `wally-train` run is interrupted and resumed (with wandb enabled),
`Trainer.train()` calls `init_wandb()` (`src/wally/training/logging.py:8`) which
invokes `wandb.init()` without a `name=` argument. W&B auto-generates a
random run name (e.g. `dainty-frost-42`), making it impossible to tell
resumed runs apart from fresh runs in the dashboard. Combined with the
sparse x-axis on resume (charts start at the resumed `global_step` rather
than 0), this hurts at-a-glance readability of long training histories.

## What Changes

- Set a deterministic wandb run name based on the wandb project and the
  starting `global_step` of the run (e.g. `wally-step-50000`). Fresh runs
  start at step 0 and produce `wally-step-0`; resumed runs produce
  `wally-step-<N>`. The name is informative in the dashboard run list
  without requiring a separate CLI flag.
- Pass the chosen `name` to `wandb.init()` from `init_wandb()` so the
  trainer's existing call site is unchanged.

## Capabilities

### New Capabilities
<!-- None — no new behavioral contract is introduced. -->

### Modified Capabilities
- `lewm-training-loop`: add a small requirement that the wandb run
  is initialized with a deterministic name of the form
  `<wandb_project>-step-<global_step>`, so resumed runs are
  identifiable in the dashboard.

## Impact

- `src/wally/training/logging.py` — `init_wandb()` gains a `name=`
  argument forwarded to `wandb.init()`.
- `src/wally/training/trainer.py` — call site at line 219 passes a name
  derived from `self.wandb_project` and `self.global_step`.
- `tests/test_train_logging.py` — add a smoke test asserting the
  generated name format.
- No CLI flags, no config-schema change, no new dependency.
- Backward compatible: existing dashboards keep working; run names just
  become predictable.
