# OpenSpec workflow

All feature work goes through OpenSpec. Config: `openspec/config.yaml`.

## Commands

- `/opsx-propose <name>` — create a change with proposal, design, specs, and tasks
- `/opsx-apply` — implement tasks from a change. Use a different subagent to implement each task.
- `/opsx-archive` — archive a completed change
- `/opsx-explore` — think through ideas before committing to a change
- `/opsx-sync` — sync delta specs from a change into main specs

## Key directories

- `openspec/specs/` — main capability specs (shared across changes, source of truth)
- `openspec/changes/` — active changes with delta specs, designs, and task lists
- `openspec/changes/archive/` — completed changes

## /opsx-apply workflow

When running `/opsx-apply`, delegate each task to a separate subagent.

Pattern for each task:
1. Read the task description and all relevant context files (specs, design, existing code)
2. Create a subagent via the Task tool with a detailed prompt covering: what to implement, which files to edit/create, how to verify (lint, typecheck, test commands), and relevant code conventions
3. The subagent returns when done — review the result, ensure tests pass, then mark the task complete in the tasks file
4. Move to the next task

Use the `general` subagent type for implementation tasks. Use the `explore` subagent type for research/investigation tasks.

Note: Use TDD (test-driven development) for tasks that contain complex logic, algorithms, or well-defined interfaces — write tests first to clarify requirements before implementation.
