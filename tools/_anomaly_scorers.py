"""Anomaly scorers + renderer for the agent anomaly contact sheet.

Pure functions: take numpy arrays, return AnomalyCluster lists (or PIL
images). No I/O. The CLI in ``tools/extract_anomalies.py`` orchestrates
loading the npz, calling these scorers, deduping the clusters, and
writing the contact sheet PNG + JSON sidecar.

This module deliberately does not import anything from ``src/wally/``.
The contact sheet is a pure offline analysis artifact.

Anomaly classes (and the agent action columns they read):
  - inv_spam      col 12 (inventory)         contiguous runs of > 0.5
  - camera_shake  cols 10, 11 (pitch, yaw)   sign-flip bursts
  - cost_spike    n/a (per-replan)           argmax of costs[]
  - attack_burst  col 7 (attack)             contiguous runs of > 0.5
  - first_event   n/a (events)               first mine_block / pickup / inv-non-empty
  - brightness    n/a (frames)               argmax / argmin of frame mean
  - best_match    n/a (frames + goal)        argmin of MSE to goal
  - final_frame   n/a (frames)               always T-1
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

INVENTORY_COL = 12
ATTACK_COL = 7
CAMERA_PITCH_COL = 10
CAMERA_YAW_COL = 11

DEFAULT_PANEL_CLASSES: tuple[str, ...] = (
    "inv_spam",
    "camera_shake",
    "cost_spike",
    "attack_burst",
    "first_event",
    "brightness",
    "best_match",
    "final_frame",
)

GRID_COLS = 4
GRID_ROWS = 2
WINDOW_RADIUS = 2
WINDOW_LEN = 2 * WINDOW_RADIUS + 1
LABEL_STRIP_HEIGHT = 32
BORDER = 8
FRAME_SCALE = 1
MIN_GAP_DEFAULT = 20


@dataclass(frozen=True)
class AnomalyCluster:
    anomaly_class: str
    center: int
    window: list[int]
    label: str
    score: float

    def __post_init__(self) -> None:
        if len(self.window) != WINDOW_LEN:
            raise ValueError(
                f"window must have length {WINDOW_LEN} (got {len(self.window)})"
            )


def _clamp_window(center: int, total: int) -> list[int]:
    """Return a ±WINDOW_RADIUS window around `center`, clamped to [0, total-1]."""
    if total <= 0:
        return []
    lo = max(0, center - WINDOW_RADIUS)
    hi = min(total - 1, center + WINDOW_RADIUS)
    indices = list(range(lo, hi + 1))
    if len(indices) < WINDOW_LEN:
        if lo == 0:
            indices = list(range(WINDOW_LEN))
        else:
            indices = list(range(total - WINDOW_LEN, total))
    return indices[:WINDOW_LEN]


def _signal_runs(
    signal: np.ndarray, threshold: float, min_length: int
) -> list[tuple[int, int]]:
    """Return [(start, end_inclusive), ...] for contiguous runs of signal>threshold.

    End is inclusive. Both ends satisfy signal>threshold.
    """
    if signal.size == 0:
        return []
    above = signal > threshold
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(above):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            if i - start >= min_length:
                runs.append((start, i - 1))
    if in_run and len(signal) - start >= min_length:
        runs.append((start, len(signal) - 1))
    return runs


def score_inv_spam(actions: np.ndarray) -> list[AnomalyCluster]:
    """One cluster per contiguous run of actions[:, 12] > 0.5 with length >= 5.

    Score = run length (longer burst = higher priority).
    """
    t = actions.shape[0]
    if t == 0:
        return []
    runs = _signal_runs(actions[:, INVENTORY_COL], threshold=0.5, min_length=5)
    out: list[AnomalyCluster] = []
    for start, end in runs:
        center = (start + end) // 2
        n = end - start + 1
        out.append(
            AnomalyCluster(
                anomaly_class="inv_spam",
                center=center,
                window=_clamp_window(center, t),
                label=f"INV SPAM t={start}..{end} ({n} steps)",
                score=float(n),
            )
        )
    return out


def score_camera_shake(actions: np.ndarray) -> list[AnomalyCluster]:
    """Bursts where pitch and yaw both sign-flip within a ±1 window for >= 4 steps."""
    t = actions.shape[0]
    if t < 3:
        return []
    pitch = actions[:, CAMERA_PITCH_COL]
    yaw = actions[:, CAMERA_YAW_COL]

    def _has_flip(signal: np.ndarray, idx: int) -> bool:
        if idx <= 0 or idx >= len(signal) - 1:
            return False
        s_left = np.sign(signal[idx - 1])
        s_self = np.sign(signal[idx])
        s_right = np.sign(signal[idx + 1])
        if 0 in (s_left, s_self, s_right):
            return False
        return (s_left != s_self) or (s_self != s_right)

    flip_mask = np.zeros(t, dtype=bool)
    for i in range(t):
        if _has_flip(pitch, i) and _has_flip(yaw, i):
            flip_mask[i] = True
    runs = _signal_runs(
        flip_mask.astype(np.float64), threshold=0.5, min_length=4
    )
    out: list[AnomalyCluster] = []
    for start, end in runs:
        center = (start + end) // 2
        n = end - start + 1
        out.append(
            AnomalyCluster(
                anomaly_class="camera_shake",
                center=center,
                window=_clamp_window(center, t),
                label=f"CAMERA SHAKE t={start}..{end} ({n} steps)",
                score=float(n),
            )
        )
    return out


def score_cost_spike(
    actions: np.ndarray, costs: np.ndarray | None
) -> list[AnomalyCluster]:
    """Single cluster at argmax(costs); [] if costs is None."""
    t = actions.shape[0]
    if costs is None or t == 0 or len(costs) == 0:
        return []
    idx = int(np.argmax(costs))
    value = float(costs[idx])
    return [
        AnomalyCluster(
            anomaly_class="cost_spike",
            center=idx,
            window=_clamp_window(idx, t),
            label=f"COST SPIKE t={idx} cost={value:.2f}",
            score=value,
        )
    ]


def score_attack_burst(actions: np.ndarray) -> list[AnomalyCluster]:
    """One cluster per contiguous run of actions[:, 7] > 0.5 with length >= 3."""
    t = actions.shape[0]
    if t == 0:
        return []
    runs = _signal_runs(actions[:, ATTACK_COL], threshold=0.5, min_length=3)
    out: list[AnomalyCluster] = []
    for start, end in runs:
        center = (start + end) // 2
        n = end - start + 1
        out.append(
            AnomalyCluster(
                anomaly_class="attack_burst",
                center=center,
                window=_clamp_window(center, t),
                label=f"ATTACK BURST t={start}..{end} ({n} steps)",
                score=float(n),
            )
        )
    return out


def score_first_event(
    actions: np.ndarray, events: np.ndarray | None
) -> list[AnomalyCluster]:
    """Single cluster at the first step with a qualifying event; [] if none."""
    t = actions.shape[0]
    if events is None or t == 0:
        return []

    def _has_event(event: Any) -> bool:
        if event is None or not isinstance(event, dict):
            return False
        mine = event.get("mine_block")
        if isinstance(mine, (list, tuple, np.ndarray)) and len(mine) > 0:
            return True
        if isinstance(mine, dict) and len(mine) > 0:
            return True
        pickup = event.get("pickup")
        if isinstance(pickup, (list, tuple, np.ndarray)) and len(pickup) > 0:
            return True
        if isinstance(pickup, dict) and len(pickup) > 0:
            return True
        inv = event.get("inventory")
        if isinstance(inv, dict):
            for slot, item in inv.items():
                if (
                    isinstance(item, dict)
                    and item.get("type", "none") != "none"
                    and item.get("quantity", 0) > 0
                ):
                    return True
        return False

    for i in range(t):
        if i < len(events) and _has_event(events[i]):
            return [
                AnomalyCluster(
                    anomaly_class="first_event",
                    center=i,
                    window=_clamp_window(i, t),
                    label=f"FIRST EVENT t={i}",
                    score=1.0,
                )
            ]
    return []


def score_brightness(actions: np.ndarray, frames: np.ndarray) -> list[AnomalyCluster]:
    """Two clusters: one at brightness.argmax() and one at brightness.argmin()."""
    t = actions.shape[0]
    if t == 0 or frames.shape[0] == 0:
        return []
    brightness = frames.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
    out: list[AnomalyCluster] = []
    i_max = int(brightness.argmax())
    out.append(
        AnomalyCluster(
            anomaly_class="brightness",
            center=i_max,
            window=_clamp_window(i_max, t),
            label=f"BRIGHTNESS MAX t={i_max} val={brightness[i_max]:.2f}",
            score=float(brightness[i_max]),
        )
    )
    i_min = int(brightness.argmin())
    if i_min != i_max:
        out.append(
            AnomalyCluster(
                anomaly_class="brightness",
                center=i_min,
                window=_clamp_window(i_min, t),
                label=f"BRIGHTNESS MIN t={i_min} val={brightness[i_min]:.2f}",
                score=float(-brightness[i_min]),
            )
        )
    return out


def score_best_match(
    actions: np.ndarray,
    frames: np.ndarray,
    goal_img: np.ndarray | None,
) -> list[AnomalyCluster]:
    """Single cluster at argmin(MSE(frames[t], goal)); [] if goal is None."""
    t = actions.shape[0]
    if goal_img is None or t == 0 or frames.shape[0] == 0:
        return []
    h, w = frames.shape[1], frames.shape[2]
    if goal_img.shape[:2] != (h, w):
        goal_pil = Image.fromarray(goal_img).resize((w, h), Image.BILINEAR)
        goal_resized = np.asarray(goal_pil, dtype=np.float32)
    else:
        goal_resized = goal_img.astype(np.float32)
    frames_f = frames.astype(np.float32)
    diffs = frames_f - goal_resized[None]
    mse = (diffs * diffs).mean(axis=(1, 2, 3))
    idx = int(mse.argmin())
    return [
        AnomalyCluster(
            anomaly_class="best_match",
            center=idx,
            window=_clamp_window(idx, t),
            label=f"BEST MATCH GOAL t={idx} mse={float(mse[idx]):.1f}",
            score=float(-mse[idx]),
        )
    ]


def score_final_frame(actions: np.ndarray) -> list[AnomalyCluster]:
    """Always one cluster at T-1."""
    t = actions.shape[0]
    if t == 0:
        return []
    return [
        AnomalyCluster(
            anomaly_class="final_frame",
            center=t - 1,
            window=_clamp_window(t - 1, t),
            label=f"FINAL FRAME t={t - 1}",
            score=0.0,
        )
    ]


def dedup_clusters(
    clusters: Sequence[AnomalyCluster], min_gap: int = MIN_GAP_DEFAULT
) -> list[AnomalyCluster]:
    """Greedy: keep higher-scored clusters first; drop any later cluster
    whose center is within `min_gap` steps of an already-kept cluster.
    Preserves input order among survivors.
    """
    indexed = sorted(enumerate(clusters), key=lambda kv: (-kv[1].score, kv[0]))
    kept_indices: list[int] = []
    kept_centers: list[int] = []
    for orig_idx, c in indexed:
        if any(abs(c.center - k) < min_gap for k in kept_centers):
            continue
        kept_indices.append(orig_idx)
        kept_centers.append(c.center)
    kept_indices.sort()
    return [clusters[i] for i in kept_indices]


def _frame_cell_size() -> tuple[int, int]:
    """Return (cell_w, cell_h) in pixels for one frame at FRAME_SCALE."""
    return 64 * FRAME_SCALE, 64 * FRAME_SCALE


def _strip_size() -> tuple[int, int]:
    """Return (strip_w, strip_h) for a 5-frame strip + label strip."""
    fw, fh = _frame_cell_size()
    return WINDOW_LEN * fw, fh + LABEL_STRIP_HEIGHT


def _grid_size() -> tuple[int, int]:
    sw, sh = _strip_size()
    return GRID_COLS * sw, GRID_ROWS * sh


def _total_size() -> tuple[int, int]:
    gw, gh = _grid_size()
    return gw + 2 * BORDER, gh + 2 * BORDER


def draw_label(image: Image.Image, x: int, y: int, width: int, text: str) -> None:
    """Draw `text` in white on a black strip at (x, y) of `width`xLABEL_STRIP_HEIGHT.

    The strip is filled with black first so existing content is overwritten.
    """
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [x, y, x + width, y + LABEL_STRIP_HEIGHT], fill=(0, 0, 0)
    )
    draw.text((x + 8, y + 6), text, fill=(255, 255, 255))


def _empty_panel(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), (0, 0, 0))


def render_panel(frames: np.ndarray, cluster: AnomalyCluster) -> Image.Image:
    """Render a single 5-frame strip + label strip for one cluster."""
    fw, fh = _frame_cell_size()
    sw, sh = _strip_size()
    panel = Image.new("RGB", (sw, sh), (0, 0, 0))
    for col, step in enumerate(cluster.window):
        if 0 <= step < frames.shape[0]:
            tile = Image.fromarray(frames[step]).resize(
                (fw, fh), Image.NEAREST
            )
            panel.paste(tile, (col * fw, 0))
    draw_label(panel, 0, fh, sw, cluster.label)
    return panel


def render_contact_sheet(
    frames: np.ndarray,
    panels: Sequence[AnomalyCluster | None],
) -> Image.Image:
    """Render a GRID_COLS x GRID_ROWS contact sheet with a BORDER-pixel
    black border. `panels` is a sequence of length <= GRID_COLS*GRID_ROWS.
    None entries (or short sequences) leave their cell black.
    """
    sw, sh = _strip_size()
    gw, gh = _grid_size()
    tw, th = _total_size()
    sheet = Image.new("RGB", (tw, th), (0, 0, 0))
    for idx, panel in enumerate(panels):
        if idx >= GRID_COLS * GRID_ROWS:
            break
        row, col = divmod(idx, GRID_COLS)
        x = BORDER + col * sw
        y = BORDER + row * sh
        if panel is None:
            continue
        rendered = render_panel(frames, panel)
        sheet.paste(rendered, (x, y))
    return sheet


def serialize_frames_json(
    npz_path: str,
    n_steps: int,
    panels: Iterable[AnomalyCluster],
    truncated: list[AnomalyCluster] | None = None,
) -> dict:
    """Build the JSON-serializable structure written to frames.json.

    `panels` are the rendered (deduped, capped) clusters. `truncated` are
    clusters dropped because the cap was exceeded; recorded separately
    so the user knows what was left out.
    """
    panel_list = list(panels)
    out: dict[str, Any] = {
        "npz": npz_path,
        "n_steps": n_steps,
        "panels": [
            {
                "panel_id": i + 1,
                "anomaly_class": c.anomaly_class,
                "label": c.label,
                "window": list(c.window),
                "score": c.score,
            }
            for i, c in enumerate(panel_list)
        ],
    }
    if truncated:
        out["truncated"] = [
            {
                "anomaly_class": c.anomaly_class,
                "label": c.label,
                "center": c.center,
                "score": c.score,
            }
            for c in truncated
        ]
    return out


def contact_sheet_size() -> tuple[int, int]:
    """Return the (width, height) of the rendered contact sheet in pixels."""
    return _total_size()
