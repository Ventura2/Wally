"""Live loss-curve dashboard for wally-train.

Parses the training log produced by ``wally-train --log-file`` and plots
prediction / sigreg / total loss against global step. Estimates the
remaining time to finish training from the recent step rate.

Usage:
    # One-shot: parse log, save PNG, print ETA to stdout.
    python tools/loss_dashboard.py --log-file logs/train.log --config configs/lewm_default.yaml --output losses.png

    # Live: redraw every 5s in a window (Ctrl-C to quit).
    python tools/loss_dashboard.py --log-file logs/train.log --config configs/lewm_default.yaml --live --interval 5

The trainer line format (from ``wally.training.trainer``) is::

    Step %d | prediction_loss=%.4f | sigreg_loss=%.4f | total_loss=%.4f | lr=%.6f

and the surrounding log record prefix is the standard ``%(asctime)s``
format used by ``wally.cli.train``. Both pieces are matched greedily
but the regex is anchored on the metric keys so unrelated lines are
ignored.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib

# Headless-safe default: only switch to an interactive backend if the
# user actually asks for --live and a display is available.
matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402


STEP_LINE_RE = re.compile(
    r"Step\s+(?P<step>\d+)\s*\|"
    r"\s*prediction_loss=(?P<pred>[-+0-9.eEInfNa]+)"
    r"\s*\|\s*sigreg_loss=(?P<sig>[-+0-9.eEInfNa]+)"
    r"\s*\|\s*total_loss=(?P<total>[-+0-9.eEInfNa]+)"
    r"\s*\|\s*lr=(?P<lr>[-+0-9.eEInfNa]+)"
)
ASCTIME_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
)


@dataclass
class StepMetrics:
    step: int
    prediction_loss: float
    sigreg_loss: float
    total_loss: float
    learning_rate: float
    timestamp: datetime | None


def _to_float(token: str) -> float:
    """Parse a metric token, accepting ``NaN``/``Inf``/``-Inf`` case-insensitively."""
    t = token.strip().lower()
    if t == "nan":
        return float("nan")
    if t in ("inf", "+inf"):
        return float("inf")
    if t == "-inf":
        return float("-inf")
    return float(token)


def _parse_asctime(line: str) -> datetime | None:
    """Return the leading ``%(asctime)s`` timestamp, if any."""
    m = ASCTIME_RE.match(line)
    if not m:
        return None
    raw = m.group("ts").replace(",", ".").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def parse_log(path: Path) -> list[StepMetrics]:
    """Read the log file and return one ``StepMetrics`` per trainer line.

    Lines that are malformed or contain non-finite values are kept so
    the plot can show gaps; only the structural parse failures are
    silently skipped.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Log file not found: {path}")
    out: list[StepMetrics] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP_LINE_RE.search(line)
            if not m:
                continue
            try:
                out.append(
                    StepMetrics(
                        step=int(m.group("step")),
                        prediction_loss=_to_float(m.group("pred")),
                        sigreg_loss=_to_float(m.group("sig")),
                        total_loss=_to_float(m.group("total")),
                        learning_rate=_to_float(m.group("lr")),
                        timestamp=_parse_asctime(line),
                    )
                )
            except ValueError:
                # Skip structurally-matching lines with unparseable floats.
                continue
    return out


def load_max_steps(config_path: Path | None) -> int | None:
    """Read ``training.max_steps`` from the YAML config, if available."""
    if config_path is None or not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return None
    training = cfg.get("training", {})
    if not isinstance(training, dict):
        return None
    value = training.get("max_steps")
    return int(value) if value is not None else None


def estimate_eta(
    metrics: list[StepMetrics], max_steps: int | None, window: int = 50
) -> tuple[float | None, timedelta | None, int | None]:
    """Estimate steps/sec, ETA, and remaining steps from the last ``window`` samples.

    Returns ``(steps_per_sec, eta, remaining)``. Any component can be
    ``None`` if it cannot be determined (no timestamps, fewer than two
    samples, or no ``max_steps``).
    """
    if not metrics:
        return None, None, None
    last = metrics[-1]
    remaining = (max_steps - last.step) if max_steps is not None else None

    # Use the last `window` samples that have a timestamp.
    ts_samples = [m for m in metrics[-window:] if m.timestamp is not None]
    if len(ts_samples) < 2:
        return None, None, remaining

    first, latest = ts_samples[0], ts_samples[-1]
    delta_t = (latest.timestamp - first.timestamp).total_seconds()
    delta_steps = latest.step - first.step
    if delta_t <= 0 or delta_steps <= 0:
        return None, None, remaining

    steps_per_sec = delta_steps / delta_t
    if remaining is None or remaining <= 0:
        return steps_per_sec, None, remaining
    eta_seconds = remaining / steps_per_sec
    return steps_per_sec, timedelta(seconds=eta_seconds), remaining


def _format_td(td: timedelta) -> str:
    """Render a ``timedelta`` as ``Hh MMm SSs`` (drops leading zero fields)."""
    total = int(td.total_seconds())
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _finite(values: Iterable[float]) -> list[float]:
    return [v for v in values if v == v and v not in (float("inf"), float("-inf"))]


def plot_losses(
    metrics: list[StepMetrics],
    max_steps: int | None,
    output: Path | None,
    show: bool,
    title: str = "Wally training loss",
) -> dict[str, object]:
    """Render the loss curves. Returns a summary dict for CLI printing."""
    if not metrics:
        raise ValueError("No training-step lines found in the log.")

    steps = [m.step for m in metrics]
    pred = _finite(m.prediction_loss for m in metrics)
    sig = _finite(m.sigreg_loss for m in metrics)
    total = _finite(m.total_loss for m in metrics)

    fig, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(steps, [m.prediction_loss for m in metrics], label="prediction_loss", linewidth=1.5)
    ax_loss.plot(steps, [m.sigreg_loss for m in metrics], label="sigreg_loss", linewidth=1.5)
    ax_loss.plot(steps, [m.total_loss for m in metrics], label="total_loss", linewidth=1.5)
    ax_loss.set_xlabel("global step")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(title)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right")

    steps_per_sec, eta, remaining = estimate_eta(metrics, max_steps)

    info_lines: list[str] = []
    info_lines.append(f"samples: {len(metrics)}")
    info_lines.append(f"current step: {metrics[-1].step}")
    if max_steps is not None:
        info_lines.append(f"max steps: {max_steps}")
        pct = 100.0 * metrics[-1].step / max_steps if max_steps else 0.0
        info_lines.append(f"progress: {pct:.2f}%")
    if steps_per_sec is not None:
        info_lines.append(f"steps/sec: {steps_per_sec:.3f}")
    if remaining is not None and remaining > 0:
        info_lines.append(f"remaining steps: {remaining}")
    if eta is not None:
        info_lines.append(f"ETA: {_format_td(eta)}")

    # ETA text overlay on the plot.
    overlay = "\n".join(info_lines)
    ax_loss.text(
        0.02,
        0.02,
        overlay,
        transform=ax_loss.transAxes,
        fontsize=9,
        family="monospace",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    # Optional secondary axis with wall-clock time.
    ts_pairs = [(m.step, m.timestamp) for m in metrics if m.timestamp is not None]
    if len(ts_pairs) >= 2:
        ax_time = ax_loss.twiny()
        ax_time.set_xlim(ts_pairs[0][1], ts_pairs[-1][1])
        ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax_time.set_xlabel("wall-clock")

    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=120)

    if show:
        # Switch to an interactive backend only when actually displaying.
        try:
            matplotlib.use("TkAgg")
        except Exception:
            try:
                matplotlib.use("Qt5Agg")
            except Exception:
                pass
        plt.show()

    plt.close(fig)

    return {
        "samples": len(metrics),
        "current_step": metrics[-1].step,
        "max_steps": max_steps,
        "steps_per_sec": steps_per_sec,
        "eta": eta,
        "remaining_steps": remaining,
        "finite_prediction": len(pred),
        "finite_sigreg": len(sig),
        "finite_total": len(total),
    }


def _print_summary(summary: dict[str, object]) -> None:
    print(f"  samples parsed : {summary['samples']}")
    print(f"  current step   : {summary['current_step']}")
    if summary.get("max_steps") is not None:
        print(f"  max steps      : {summary['max_steps']}")
    if summary.get("steps_per_sec") is not None:
        print(f"  steps/sec      : {summary['steps_per_sec']:.3f}")
    if summary.get("remaining_steps") is not None and summary["remaining_steps"] > 0:
        print(f"  remaining      : {summary['remaining_steps']} steps")
    if summary.get("eta") is not None:
        print(f"  ETA            : {_format_td(summary['eta'])}")
    missing = []
    for key, label in (
        ("finite_prediction", "prediction_loss"),
        ("finite_sigreg", "sigreg_loss"),
        ("finite_total", "total_loss"),
    ):
        if summary.get(key) != summary.get("samples"):
            missing.append(label)
    if missing:
        print(f"  non-finite     : {', '.join(missing)} (kept as gaps in the plot)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss curves and ETA for a wally-train run.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Path to the training log file (the one passed to wally-train --log-file).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the YAML training config; used to read training.max_steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps (otherwise read from --config).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save the plot to this PNG path. Default: losses.png next to the log.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Tail the log and redraw the plot every --interval seconds.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds for --live mode (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    max_steps = args.max_steps
    if max_steps is None:
        max_steps = load_max_steps(args.config)

    if args.live:
        output = args.output  # may be None; will save per-tick into a live path
        live_output = output or args.log_file.with_name("losses_live.png")
        matplotlib.use("TkAgg")  # best-effort interactive backend
        try:
            plt.ion()
        except Exception:
            pass
        last_size = -1
        try:
            while True:
                if args.log_file.is_file():
                    size = args.log_file.stat().st_size
                    if size != last_size:
                        last_size = size
                        metrics = parse_log(args.log_file)
                        if metrics:
                            print(
                                f"[{datetime.now():%H:%M:%S}] step {metrics[-1].step}"
                                f" / {max_steps if max_steps else '?'}"
                            )
                            summary = plot_losses(
                                metrics,
                                max_steps,
                                live_output,
                                show=True,
                            )
                            _print_summary(summary)
                            plt.pause(0.001)
                time.sleep(max(0.5, args.interval))
        except KeyboardInterrupt:
            print("\nStopped.")
        return 0

    # One-shot mode.
    metrics = parse_log(args.log_file)
    if not metrics:
        print(f"No training-step lines found in {args.log_file}", file=sys.stderr)
        return 1
    output = args.output or args.log_file.with_name("losses.png")
    summary = plot_losses(metrics, max_steps, output, show=False)
    print(f"Saved plot to {output}")
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
