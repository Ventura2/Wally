# src/collector — MineStudio data collection

This subpackage owns trajectory collection. **It only runs inside the `wally-dev` Podman container** because MineStudio's Java engine + LWJGL natives are Linux-only.

## Files

- `env.py` — MineStudio simulator wrapper (heads-up display, action space, frame extractor)
- `buffer.py` — per-episode step buffer
- `recorder.py` — per-step JPEG frame + JSON action sidecar writer
- `config.py` — collector config (task, max-steps, output dir)
- `raw_shard_writer.py` — WebDataset `.tar` shard writer

## Container quirks

- Uses **system Python 3.10** and system-installed `minestudio` at `/usr/local/lib/python3.10/dist-packages/`
- Requires `PYTHONPATH=src` because the `src/`-layout package isn't installed via pip in the container
- The Minecraft engine fat jar is at `/tmp/MineStudio/engine/build/libs/mcprec-6.13.jar` (downloaded by `python -m minestudio.simulator.entry -y`)
- A symlink from `MCP-Reborn/build/libs/mcprec-6.13.jar` → engine jar exists at `/workspace/.venv/Lib/site-packages/minestudio/simulator/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar`
- Known benign warnings: `fliteWrapper` library, `optifine/ctm/default/empty.png` texture, OpenAL sound device, Realms auth — all safe to ignore

## Run command

```sh
# Inside the container (wally-dev)
podman exec wally-dev sh -c 'cd /workspace && PYTHONPATH=src python3 -m wally.cli.collect --episodes 1 --output-dir data/raw --max-steps 5000'
```

`--max-steps` prevents infinite episodes (the `HumanSurvival` task only ends on player death, which may never happen with random actions). Always pass it.

## Data flow

- **Raw shards** (`data/raw/*.tar`): per-step JPEG frames + JSON action sidecars
- Convert to **training shards** (`data/shards/*.tar`) via `wally-convert` before training

The training-side sharding + format is documented in `src/wally/AGENTS.md`.
