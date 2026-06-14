## Context

`Trainer.train()` (`src/wally/training/trainer.py:216`) calls
`init_wandb(self.config)` at line 219. The current implementation in
`src/wally/training/logging.py:8` is a thin wrapper around
`wandb.init(project=project_name, config=config)` with no `name=`
argument, so W&B auto-assigns a random display name. On resume, the
new run is a separate row in the dashboard with a sparse x-axis
starting at the resumed `global_step`; without a meaningful name it
is hard to tell at a glance which run is a continuation of which.

## Goals / Non-Goals

**Goals:**
- Set a deterministic, human-readable wandb run name derived from the
  project and starting `global_step`.
- Single call site change, no new CLI flags, no config-schema change.

**Non-Goals:**
- Reusing the previous run's `id=` (would require
  `resume="allow"` and persisting the run id in the checkpoint —
  intentionally out of scope for this small change).
- Backfilling the sparse x-axis with synthetic points.

## Decisions

- **Name format: `<wandb_project>-step-<global_step>`** — short,
  sortable in the dashboard list, and unambiguous. Alternatives
  considered:
  - `resume-{N}` suffix only: hides that the project is `wally`,
    collides across multiple projects.
  - Timestamp: not sortable against the step, less useful for
    correlating with checkpoint files.
  - UUID: defeats the purpose (same as the current auto-name).
- **Compute the name at the call site, not inside `init_wandb`** —
  `init_wandb` is a generic helper. The trainer already knows its
  `wandb_project` and `global_step`, so it is the natural owner of
  the naming policy. The helper still accepts a `name=` kwarg and
  forwards it to `wandb.init()`.
- **Read `wandb_project` from `self.config`** (matches what
  `init_wandb` already does) rather than threading a new attribute
  through `__init__`.

## Risks / Trade-offs

- [Two resumed runs at the same `global_step` collide in the
  dashboard] → W&B appends a numeric suffix automatically, so the
  chart is still legible. Acceptable.
- [Existing dashboards that filter by run name break] → run names
  change from `dainty-frost-42` to `wally-step-N`. Mitigated by
  being a one-time transition; no programmatic dashboard filters
  known to depend on the old name.

## Migration Plan

- No data migration. Old wandb runs are untouched.
- New runs adopt the deterministic name immediately on deploy.

## Open Questions

None.
