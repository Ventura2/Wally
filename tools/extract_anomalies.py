"""Extract anomaly contact sheet from a recorded episode npz.

Reads ``episode_0.npz`` (or any compatible npz with the same keys),
runs the per-step anomaly scorers in ``tools/_anomaly_scorers.py``,
deduplicates nearby clusters, and writes a single labeled contact
sheet (PNG) plus a JSON sidecar describing each panel.

Usage:
    python tools/extract_anomalies.py <npz> [--goal-frame PATH]
        [--panels CSV] [--out-png PATH] [--out-json PATH]

By default the PNG and JSON are written next to the input npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _anomaly_scorers import (  # noqa: E402
    DEFAULT_PANEL_CLASSES,
    GRID_COLS,
    GRID_ROWS,
    AnomalyCluster,
    dedup_clusters,
    render_contact_sheet,
    score_attack_burst,
    score_best_match,
    score_brightness,
    score_camera_shake,
    score_cost_spike,
    score_final_frame,
    score_first_event,
    score_inv_spam,
    serialize_frames_json,
)


def _scorer_inv_spam(actions, frames, costs, events, goal):
    return score_inv_spam(actions)


def _scorer_camera_shake(actions, frames, costs, events, goal):
    return score_camera_shake(actions)


def _scorer_cost_spike(actions, frames, costs, events, goal):
    return score_cost_spike(actions, costs)


def _scorer_attack_burst(actions, frames, costs, events, goal):
    return score_attack_burst(actions)


def _scorer_first_event(actions, frames, costs, events, goal):
    return score_first_event(actions, events)


def _scorer_brightness(actions, frames, costs, events, goal):
    return score_brightness(actions, frames)


def _scorer_best_match(actions, frames, costs, events, goal):
    return score_best_match(actions, frames, goal)


def _scorer_final_frame(actions, frames, costs, events, goal):
    return score_final_frame(actions)


PANEL_SCORERS: dict[str, callable] = {
    "inv_spam": _scorer_inv_spam,
    "camera_shake": _scorer_camera_shake,
    "cost_spike": _scorer_cost_spike,
    "attack_burst": _scorer_attack_burst,
    "first_event": _scorer_first_event,
    "brightness": _scorer_brightness,
    "best_match": _scorer_best_match,
    "final_frame": _scorer_final_frame,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an anomaly contact sheet from a wally agent npz."
    )
    parser.add_argument(
        "npz",
        type=Path,
        help=(
            "Path to episode_0.npz (or any npz with `frames` and `actions`)."
        ),
    )
    parser.add_argument(
        "--goal-frame",
        type=Path,
        default=None,
        help="Path to a goal PNG for the best-match-to-goal panel. Optional.",
    )
    parser.add_argument(
        "--panels",
        type=str,
        default=",".join(DEFAULT_PANEL_CLASSES),
        help=(
            "Comma-separated list of anomaly classes to render. "
            f"Available: {','.join(DEFAULT_PANEL_CLASSES)}. "
            "Default: all of them (capped at 8 in the rendered grid)."
        ),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Output PNG path. Default: <npz_dir>/anomaly_contact_sheet.png",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON sidecar path. Default: <npz_dir>/frames.json",
    )
    return parser.parse_args(argv)


def _load_npz(npz_path: Path) -> dict:
    if not npz_path.exists():
        print(f"ERROR: npz not found: {npz_path}", file=sys.stderr)
        sys.exit(2)
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: failed to load npz {npz_path}: {e}", file=sys.stderr)
        sys.exit(2)
    for key in ("frames", "actions"):
        if key not in data.files:
            print(f"ERROR: npz missing required key '{key}'", file=sys.stderr)
            sys.exit(2)
    return data


def _load_goal(goal_path: Path | None) -> np.ndarray | None:
    if goal_path is None:
        return None
    if not goal_path.exists():
        print(f"ERROR: goal frame not found: {goal_path}", file=sys.stderr)
        sys.exit(2)
    return np.asarray(Image.open(goal_path).convert("RGB"), dtype=np.uint8)


def _resolve_panels_arg(panels_arg: str) -> list[str]:
    parts = [p.strip() for p in panels_arg.split(",") if p.strip()]
    if not parts:
        print("ERROR: --panels must list at least one anomaly class", file=sys.stderr)
        sys.exit(2)
    unknown = [p for p in parts if p not in PANEL_SCORERS]
    if unknown:
        print(
            f"ERROR: unknown panel class '{unknown[0]}'; "
            f"available: {','.join(PANEL_SCORERS)}",
            file=sys.stderr,
        )
        sys.exit(2)
    return parts


def _collect_clusters(
    panel_classes: list[str],
    actions: np.ndarray,
    frames: np.ndarray,
    costs: np.ndarray | None,
    events: np.ndarray | None,
    goal_img: np.ndarray | None,
) -> tuple[list[AnomalyCluster], list[AnomalyCluster]]:
    """Run the requested scorers, dedup, and cap at GRID_COLS*GRID_ROWS.

    Returns (rendered_panels, truncated_dropped).
    """
    collected: list[AnomalyCluster] = []
    for cls in panel_classes:
        collected.extend(
            PANEL_SCORERS[cls](actions, frames, costs, events, goal_img)
        )
    collected = dedup_clusters(collected)
    cap = GRID_COLS * GRID_ROWS
    if len(collected) > cap:
        truncated = collected[cap:]
        collected = collected[:cap]
        print(
            f"warning: {len(truncated)} panel(s) truncated to fit the "
            f"{GRID_COLS}x{GRID_ROWS} grid (use --panels to choose a subset)",
            file=sys.stderr,
        )
        return collected, truncated
    return collected, []


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    npz_path: Path = args.npz.resolve()
    data = _load_npz(npz_path)
    frames = data["frames"]
    actions = data["actions"]
    costs = data.get("costs") if "costs" in data.files else None
    events = data.get("events") if "events" in data.files else None
    goal_img = _load_goal(args.goal_frame)
    panel_classes = _resolve_panels_arg(args.panels)

    rendered, truncated = _collect_clusters(
        panel_classes, actions, frames, costs, events, goal_img
    )

    n_steps = int(frames.shape[0])
    img = render_contact_sheet(frames, rendered)

    npz_dir = npz_path.parent
    out_png = (
        args.out_png
        if args.out_png is not None
        else npz_dir / "anomaly_contact_sheet.png"
    )
    out_json = (
        args.out_json
        if args.out_json is not None
        else npz_dir / "frames.json"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    img.save(out_png)
    payload = serialize_frames_json(
        str(npz_path), n_steps, rendered, truncated=truncated
    )
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"wrote {out_png}  ({img.size[0]}x{img.size[1]}, {len(rendered)} panels)")
    print(f"wrote {out_json}  ({len(rendered)} panels, {n_steps} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
