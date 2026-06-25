"""Analyze a wally agent trajectory npz for task-completion evidence.

Usage:
    python tools/analyze_trajectory.py <path/to/episode_0.npz>

Looks at:
- events[*] for inventory changes, pickups, mine_block, break_item
- actions[*] for movement patterns (forward, attack, inventory spam)
- frames metadata (size, count, brightness changes)

Verdict section at the bottom tells you whether the agent did anything useful.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ACTION_LABELS = [
    "forward", "backward", "left", "right", "jump", "sneak", "sprint",
    "attack", "use", "drop", "camera_pitch", "camera_yaw",
    "inventory", "hotbar_1", "hotbar_2", "hotbar_3", "hotbar_4", "hotbar_5",
    "hotbar_6", "hotbar_7", "hotbar_8", "hotbar_9", "pickItem", "placeItem", "craft",
]

WOOD_KEYWORDS = (
    "log", "wood", "oak", "birch", "spruce", "jungle", "acacia",
    "dark_oak", "mangrove", "cherry",
)


def main() -> None:
    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"D:\Projects\Personal\artificial-intelligence\wally\ag-tests\run_wood\episode_0.npz"
    )

    print(f"Analyzing: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    print(f"Keys: {list(data.keys())}")

    frames = data["frames"]
    actions = data["actions"]
    events = data.get("events")
    print(f"frames:  shape={frames.shape} dtype={frames.dtype}")
    print(f"actions: shape={actions.shape} dtype={actions.dtype}")
    if events is not None:
        print(f"events:  shape={events.shape} dtype={events.dtype}")
    else:
        print("events: <none>")
    print()

    print("=== Action summary (mean over time) ===")
    print(f"{'action':>14}  {'mean':>8}  {'max':>8}  {'non-zero':>10}")
    for i, name in enumerate(ACTION_LABELS[: actions.shape[1]]):
        col = actions[:, i]
        print(f"{name:>14}  {col.mean():>8.3f}  {np.abs(col).max():>8.3f}  {(np.abs(col) > 0.01).sum():>10}")
    print()

    print("=== Attack (col 7) ===")
    attack = actions[:, 7]
    print(f"  attack > 0.5: {(attack > 0.5).sum()}")
    print(f"  attack > 0.1: {(attack > 0.1).sum()}")

    print()
    print("=== Movement (cols 0-3) ===")
    for i, name in enumerate(["forward", "backward", "left", "right"]):
        col = actions[:, i]
        print(f"  {name:>10} > 0.5 for {(col > 0.5).sum()} steps")

    print()
    print("=== Camera (cols 10-11) ===")
    pitch = actions[:, 10]
    yaw = actions[:, 11]
    print(f"  pitch range: [{pitch.min():.2f}, {pitch.max():.2f}], total |delta| > 0.1: {(np.abs(np.diff(pitch)) > 0.1).sum()}")
    print(f"  yaw range:   [{yaw.min():.2f}, {yaw.max():.2f}], total |delta| > 0.1: {(np.abs(np.diff(yaw)) > 0.1).sum()}")

    print()
    print("=== Inventory action (col 12) ===")
    inv = actions[:, 12]
    print(f"  inv > 0.5: {(inv > 0.5).sum()} (the CEM local-minimum signal)")
    print(f"  inv > 0.1: {(inv > 0.1).sum()}")

    if events is not None and len(events) > 0:
        non_null = [e for e in events if e is not None]
        print()
        print("=== Events (non-null) ===")
        print(f"Total non-null events: {len(non_null)} of {len(events)}")

        # Look for non-empty pickup/mine_block/break_item across all events
        pickup_total = 0
        mine_total = 0
        break_total = 0
        inventory_nonempty_steps = 0
        for e in events:
            if e is None:
                continue
            if e.get("pickup") and len(e["pickup"]) > 0:
                pickup_total += 1
            if e.get("mine_block") and len(e["mine_block"]) > 0:
                mine_total += 1
            if e.get("break_item") and len(e["break_item"]) > 0:
                break_total += 1
            inv = e.get("inventory", {})
            if isinstance(inv, dict):
                for slot, item in inv.items():
                    if isinstance(item, dict) and item.get("type", "none") != "none" and item.get("quantity", 0) > 0:
                        inventory_nonempty_steps += 1
                        break
        print(f"Steps with non-empty pickup:       {pickup_total}")
        print(f"Steps with non-empty mine_block:   {mine_total}")
        print(f"Steps with non-empty break_item:   {break_total}")
        print(f"Steps with at least one item held: {inventory_nonempty_steps}")

        print()
        print(f"=== Wood-related events (keywords: {WOOD_KEYWORDS}) ===")
        found = []
        for i, e in enumerate(events):
            if e is None:
                continue
            e_str = str(e).lower()
            for kw in WOOD_KEYWORDS:
                if kw in e_str:
                    found.append((i, e))
                    break
        print(f"Wood-related events found: {len(found)}")
        for i, e in found[:10]:
            print(f"  step {i}: {e}")
    else:
        print()
        print("=== Events: none ===")

    print()
    print("=== Frames: sample stats ===")
    print(f"  mean brightness: {frames.mean():.1f}")
    print(f"  per-step mean brightness (first 5): {frames[:5].mean(axis=(1, 2, 3))}")
    print(f"  per-step mean brightness (last 5):  {frames[-5:].mean(axis=(1, 2, 3))}")
    deltas = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
    print(f"  per-step |diff| first 5:  {deltas[:5]}")
    print(f"  per-step |diff| middle 5: {deltas[len(deltas)//2 - 2 : len(deltas)//2 + 3]}")
    print(f"  per-step |diff| last 5:   {deltas[-5:]}")

    print()
    print("=== Verdict ===")
    if events is not None:
        if any(
            (e is not None and e.get("inventory") and any(
                isinstance(s, dict) and s.get("type", "none") != "none" and s.get("quantity", 0) > 0
                for s in e["inventory"].values()
            ))
            for e in events
        ):
            print("  SUCCESS: agent held at least one item at some point")
        else:
            inv_high = (actions[:, 12] > 0.5).sum()
            mine = sum(1 for e in events if e is not None and e.get("mine_block"))
            pickup = sum(1 for e in events if e is not None and e.get("pickup"))
            print(f"  FAIL: agent never held any item")
            print(f"    inventory spam (col 12 > 0.5): {inv_high} steps")
            print(f"    steps with mine_block event:   {mine}")
            print(f"    steps with pickup event:       {pickup}")
            if inv_high > 50:
                print("    >> CEM local minimum detected — agent stuck opening inventory")
                print("    >> Fix: either train more, or apply action[12]=0 mask in src/wally/agent/loop.py")


if __name__ == "__main__":
    main()
