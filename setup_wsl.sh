#!/bin/bash
set -e

cd ~/wally
source .venv/bin/activate

echo "=== Installing project dependencies ==="
uv sync 2>&1

echo "=== Installing minestudio ==="
uv pip install minestudio 2>&1

echo "=== Running tests ==="
uv run pytest -v 2>&1

echo "=== Done ==="
