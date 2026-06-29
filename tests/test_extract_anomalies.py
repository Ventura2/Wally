"""Tests for tools/_anomaly_scorers.py and tools/extract_anomalies.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

# ruff: noqa: E402,I001  (intentional: tools/_*.py path injection)
import extract_anomalies as cli
from _anomaly_scorers import (
    AnomalyCluster,
    DEFAULT_PANEL_CLASSES,
    contact_sheet_size,
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


@pytest.fixture
def make_fake_npz(tmp_path: Path):
    """Factory: write a 64x64x3 frames array and a 25-dim action array to an npz.

    Optional kwargs: ``n_steps`` (default 100), ``costs`` (1D array or None),
    ``events`` (1D object array or None), ``inv_bursts`` (list of (start,end)
    inclusive tuples that set actions[:, 12] > 0.5 in those ranges),
    ``attack_bursts`` (same for actions[:, 7]), ``camera_flip_bursts``
    (same for both pitch (col 10) and yaw (col 11) set to alternating ±1).
    """

    def _make(
        n_steps: int = 100,
        *,
        costs: np.ndarray | None = None,
        events: np.ndarray | None = None,
        inv_bursts: list[tuple[int, int]] | None = None,
        attack_bursts: list[tuple[int, int]] | None = None,
        camera_flip_bursts: list[tuple[int, int]] | None = None,
    ) -> Path:
        frames = np.random.randint(0, 256, size=(n_steps, 64, 64, 3), dtype=np.uint8)
        actions = np.zeros((n_steps, 25), dtype=np.float32)
        for start, end in inv_bursts or []:
            actions[start : end + 1, 12] = 1.0
        for start, end in attack_bursts or []:
            actions[start : end + 1, 7] = 1.0
        for start, end in camera_flip_bursts or []:
            actions[start : end + 1, 10] = np.where(
                np.arange(start, end + 1) % 2 == 0, 1.0, -1.0
            )
            actions[start : end + 1, 11] = np.where(
                np.arange(start, end + 1) % 2 == 0, 1.0, -1.0
            )
        kwargs: dict = {"frames": frames, "actions": actions}
        if costs is not None:
            kwargs["costs"] = costs
        if events is not None:
            kwargs["events"] = events
        path = tmp_path / "fake.npz"
        np.savez(path, **kwargs)
        return path

    return _make


# --- Scorer tests ---------------------------------------------------------


def test_score_inv_spam_detects_known_burst(make_fake_npz):
    path = make_fake_npz(inv_bursts=[(20, 30)])
    data = np.load(path, allow_pickle=True)
    clusters = score_inv_spam(data["actions"])
    assert len(clusters) == 1
    assert clusters[0].anomaly_class == "inv_spam"
    assert clusters[0].center == 25
    assert clusters[0].window == [23, 24, 25, 26, 27]
    assert "t=20..30" in clusters[0].label
    assert "(11 steps)" in clusters[0].label


def test_score_inv_spam_ignores_short_bursts(make_fake_npz):
    path = make_fake_npz(inv_bursts=[(20, 22)])
    data = np.load(path, allow_pickle=True)
    clusters = score_inv_spam(data["actions"])
    assert clusters == []


def test_score_camera_shake_detects(make_fake_npz):
    # The flip detector only flags steps with both neighbors, so a 50..60
    # burst produces a flip run of 51..59 (length 9).
    path = make_fake_npz(camera_flip_bursts=[(50, 60)])
    data = np.load(path, allow_pickle=True)
    clusters = score_camera_shake(data["actions"])
    assert len(clusters) == 1
    assert clusters[0].anomaly_class == "camera_shake"
    assert "t=51..59" in clusters[0].label


def test_score_cost_spike_at_argmax(make_fake_npz):
    costs = np.linspace(0, 1, 100)
    costs[77] = 9.5
    path = make_fake_npz(costs=costs)
    data = np.load(path, allow_pickle=True)
    clusters = score_cost_spike(data["actions"], data["costs"])
    assert len(clusters) == 1
    assert clusters[0].center == 77
    assert clusters[0].score == 9.5
    assert "cost=9.50" in clusters[0].label


def test_score_cost_spike_skipped_when_missing(make_fake_npz):
    path = make_fake_npz()
    data = np.load(path, allow_pickle=True)
    assert score_cost_spike(data["actions"], None) == []


def test_score_attack_burst_detects(make_fake_npz):
    path = make_fake_npz(attack_bursts=[(30, 35)])
    data = np.load(path, allow_pickle=True)
    clusters = score_attack_burst(data["actions"])
    assert len(clusters) == 1
    assert clusters[0].anomaly_class == "attack_burst"
    assert clusters[0].center == 32
    assert "t=30..35" in clusters[0].label


def test_score_first_event_returns_first(make_fake_npz):
    events = np.empty(100, dtype=object)
    events[42] = {"mine_block": [{"type": "log"}]}
    path = make_fake_npz(events=events)
    data = np.load(path, allow_pickle=True)
    clusters = score_first_event(data["actions"], data["events"])
    assert len(clusters) == 1
    assert clusters[0].center == 42
    assert clusters[0].anomaly_class == "first_event"


def test_score_first_event_handles_pickup(make_fake_npz):
    events = np.empty(100, dtype=object)
    events[50] = {"pickup": [{"type": "log"}]}
    path = make_fake_npz(events=events)
    data = np.load(path, allow_pickle=True)
    clusters = score_first_event(data["actions"], data["events"])
    assert clusters[0].center == 50


def test_score_first_event_handles_inventory_nonempty(make_fake_npz):
    events = np.empty(100, dtype=object)
    events[50] = {"inventory": {"0": {"type": "log", "quantity": 1}}}
    path = make_fake_npz(events=events)
    data = np.load(path, allow_pickle=True)
    clusters = score_first_event(data["actions"], data["events"])
    assert clusters[0].center == 50


def test_score_first_event_none_when_no_events(make_fake_npz):
    path = make_fake_npz(events=np.empty(100, dtype=object))
    data = np.load(path, allow_pickle=True)
    assert score_first_event(data["actions"], data["events"]) == []


def test_score_brightness_returns_two(make_fake_npz):
    path = make_fake_npz(n_steps=50)
    data = np.load(path, allow_pickle=True)
    data["frames"][10] = 0
    data["frames"][20] = 255
    np.savez(path, **dict(data))
    data = np.load(path, allow_pickle=True)
    clusters = score_brightness(data["actions"], data["frames"])
    assert len(clusters) == 2
    assert {c.label.split()[1] for c in clusters} == {"MAX", "MIN"}


def test_score_best_match_with_goal():
    frames = np.zeros((20, 64, 64, 3), dtype=np.uint8)
    frames[5] = 200
    goal = np.full((64, 64, 3), 200, dtype=np.uint8)
    actions = np.zeros((20, 25), dtype=np.float32)
    clusters = score_best_match(actions, frames, goal)
    assert len(clusters) == 1
    assert clusters[0].center == 5
    assert "mse=0.0" in clusters[0].label


def test_score_best_match_without_goal():
    frames = np.zeros((20, 64, 64, 3), dtype=np.uint8)
    actions = np.zeros((20, 25), dtype=np.float32)
    assert score_best_match(actions, frames, None) == []


def test_score_final_frame_always_present():
    actions = np.zeros((42, 25), dtype=np.float32)
    clusters = score_final_frame(actions)
    assert len(clusters) == 1
    assert clusters[0].center == 41
    assert "t=41" in clusters[0].label


def test_window_clamps_at_edges():
    actions = np.zeros((10, 25), dtype=np.float32)
    # Burst at steps 0..4 (length 5). Center is 2, window = [0,1,2,3,4].
    actions[0:5, 12] = 1.0
    clusters = score_inv_spam(actions)
    assert len(clusters) == 1
    assert clusters[0].center == 2
    assert clusters[0].window == [0, 1, 2, 3, 4]


# --- Dedup tests ----------------------------------------------------------


def test_dedup_merges_close_clusters():
    c1 = AnomalyCluster("inv_spam", 55, [53, 54, 55, 56, 57], "A", 10.0)
    c2 = AnomalyCluster("inv_spam", 67, [65, 66, 67, 68, 69], "B", 5.0)
    kept = dedup_clusters([c1, c2], min_gap=20)
    assert kept == [c1]


def test_dedup_keeps_far_apart():
    c1 = AnomalyCluster("inv_spam", 55, [53, 54, 55, 56, 57], "A", 10.0)
    c2 = AnomalyCluster("cost_spike", 400, [398, 399, 400, 401, 402], "B", 0.5)
    kept = dedup_clusters([c1, c2], min_gap=20)
    assert len(kept) == 2
    assert c1 in kept and c2 in kept


def test_dedup_higher_score_wins():
    c_low = AnomalyCluster("inv_spam", 55, [53, 54, 55, 56, 57], "A", 1.0)
    c_high = AnomalyCluster("cost_spike", 60, [58, 59, 60, 61, 62], "B", 9.0)
    kept = dedup_clusters([c_low, c_high], min_gap=20)
    assert kept == [c_high]


# --- Renderer tests -------------------------------------------------------


def test_render_produces_4x2_grid():
    frames = np.random.randint(0, 256, size=(50, 64, 64, 3), dtype=np.uint8)
    panels = [
        AnomalyCluster("inv_spam", 10, [8, 9, 10, 11, 12], "A", 1.0),
        AnomalyCluster("inv_spam", 20, [18, 19, 20, 21, 22], "B", 1.0),
        AnomalyCluster("inv_spam", 30, [28, 29, 30, 31, 32], "C", 1.0),
        AnomalyCluster("inv_spam", 40, [38, 39, 40, 41, 42], "D", 1.0),
        AnomalyCluster("inv_spam", 12, [10, 11, 12, 13, 14], "E", 1.0),
        AnomalyCluster("inv_spam", 22, [20, 21, 22, 23, 24], "F", 1.0),
        AnomalyCluster("inv_spam", 32, [30, 31, 32, 33, 34], "G", 1.0),
        AnomalyCluster("inv_spam", 42, [40, 41, 42, 43, 44], "H", 1.0),
    ]
    img = render_contact_sheet(frames, panels)
    assert img.size == (1296, 208)
    assert contact_sheet_size() == (1296, 208)


def test_render_handles_fewer_panels():
    frames = np.random.randint(0, 256, size=(50, 64, 64, 3), dtype=np.uint8)
    panels = [
        AnomalyCluster("final_frame", 49, [47, 48, 49, 48, 49], "F", 0.0)
    ]
    img = render_contact_sheet(frames, panels)
    assert img.size == (1296, 208)


def test_serialize_frames_json_shape():
    panels = [
        AnomalyCluster("inv_spam", 10, [8, 9, 10, 11, 12], "INV", 5.0),
        AnomalyCluster("cost_spike", 40, [38, 39, 40, 41, 42], "COST", 0.7),
    ]
    payload = serialize_frames_json("/tmp/x.npz", 100, panels)
    assert payload["npz"] == "/tmp/x.npz"
    assert payload["n_steps"] == 100
    assert len(payload["panels"]) == 2
    assert payload["panels"][0]["window"] == [8, 9, 10, 11, 12]
    assert payload["panels"][0]["anomaly_class"] == "inv_spam"
    assert payload["panels"][0]["score"] == 5.0
    assert payload["panels"][1]["score"] == 0.7
    assert "truncated" not in payload


def test_serialize_frames_json_with_truncated():
    panels = [AnomalyCluster("inv_spam", 10, [8, 9, 10, 11, 12], "A", 5.0)]
    truncated = [AnomalyCluster("cost_spike", 100, [98, 99, 100, 99, 100], "B", 0.1)]
    payload = serialize_frames_json("/x.npz", 200, panels, truncated=truncated)
    assert "truncated" in payload
    assert payload["truncated"][0]["anomaly_class"] == "cost_spike"


# --- CLI tests ------------------------------------------------------------


def test_missing_npz_exits_2(tmp_path, capsys):
    bogus = tmp_path / "does-not-exist.npz"
    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(bogus)])
    assert exc_info.value.code == 2
    assert "ERROR: npz not found" in capsys.readouterr().err


def test_missing_required_key_exits_2(tmp_path, capsys):
    bad = tmp_path / "bad.npz"
    np.savez(bad, frames=np.zeros((10, 64, 64, 3), dtype=np.uint8))
    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(bad)])
    assert exc_info.value.code == 2
    assert "ERROR: npz missing required key 'actions'" in capsys.readouterr().err


def test_unknown_panel_exits_2(make_fake_npz, capsys):
    path = make_fake_npz()
    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(path), "--panels", "inv_spam,foo"])
    assert exc_info.value.code == 2
    assert "unknown panel class 'foo'" in capsys.readouterr().err


def test_empty_panels_exits_2(make_fake_npz, capsys):
    path = make_fake_npz()
    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(path), "--panels", ","])
    assert exc_info.value.code == 2
    assert "must list at least one" in capsys.readouterr().err


def test_panels_flag_filters_output(make_fake_npz, tmp_path):
    path = make_fake_npz(
        inv_bursts=[(10, 20)],
        attack_bursts=[(40, 45)],
    )
    out_png = tmp_path / "out.png"
    out_json = tmp_path / "out.json"
    cli.main(
        [
            str(path),
            "--panels",
            "inv_spam,attack_burst",
            "--out-png",
            str(out_png),
            "--out-json",
            str(out_json),
        ]
    )
    assert out_png.exists()
    payload = json.loads(out_json.read_text())
    assert len(payload["panels"]) == 2
    classes = {p["anomaly_class"] for p in payload["panels"]}
    assert classes == {"inv_spam", "attack_burst"}


def test_goal_frame_omitted_when_not_provided(make_fake_npz, tmp_path):
    path = make_fake_npz()
    out_json = tmp_path / "out.json"
    cli.main(
        [
            str(path),
            "--panels",
            "best_match",
            "--out-json",
            str(out_json),
        ]
    )
    payload = json.loads(out_json.read_text())
    assert all(p["anomaly_class"] != "best_match" for p in payload["panels"])


def test_default_output_paths_next_to_npz(make_fake_npz):
    path = make_fake_npz()
    cli.main([str(path)])
    assert (path.parent / "anomaly_contact_sheet.png").exists()
    assert (path.parent / "frames.json").exists()


def test_custom_output_paths(make_fake_npz, tmp_path):
    path = make_fake_npz()
    out_png = tmp_path / "custom.png"
    out_json = tmp_path / "custom.json"
    cli.main(
        [str(path), "--out-png", str(out_png), "--out-json", str(out_json)]
    )
    assert out_png.exists()
    assert out_json.exists()
    assert not (path.parent / "anomaly_contact_sheet.png").exists()


def test_no_side_effects_on_input(make_fake_npz):
    path = make_fake_npz()
    mtime_before = path.stat().st_mtime
    size_before = path.stat().st_size
    cli.main([str(path), "--out-png", str(path.parent / "x.png")])
    assert path.stat().st_mtime == mtime_before
    assert path.stat().st_size == size_before


def test_default_panel_classes_match_scorers():
    assert set(DEFAULT_PANEL_CLASSES) == set(cli.PANEL_SCORERS.keys())


@pytest.mark.smoke
def test_cli_end_to_end_on_fake(make_fake_npz, tmp_path):
    # Place inv_spam, attack_burst, and cost_spike far enough apart that
    # the min_gap=20 dedup keeps all of them. Brightness extremes land at
    # controlled positions (steps 0 = all 0, step 119 = all 255) so they
    # don't overlap with the action bursts.
    n = 120
    frames = np.full((n, 64, 64, 3), 128, dtype=np.uint8)
    # Brightness extremes at well-separated positions so dedup keeps them.
    frames[0] = 0
    frames[50] = 255
    costs = np.linspace(0, 1, n)
    # Cost spike at step 90 (far enough from final_frame at 119 to survive
    # the min_gap=20 dedup).
    costs[90] = 5.0
    path = tmp_path / "e2e.npz"
    actions = np.zeros((n, 25), dtype=np.float32)
    actions[20:31, 12] = 1.0
    actions[60:71, 7] = 1.0
    np.savez(path, frames=frames, actions=actions, costs=costs)
    out_png = tmp_path / "e2e.png"
    out_json = tmp_path / "e2e.json"
    cli.main([str(path), "--out-png", str(out_png), "--out-json", str(out_json)])
    img = Image.open(out_png)
    assert img.size == (1296, 208)
    payload = json.loads(out_json.read_text())
    assert payload["n_steps"] == n
    classes = {p["anomaly_class"] for p in payload["panels"]}
    assert "inv_spam" in classes
    assert "attack_burst" in classes
    assert "cost_spike" in classes
    assert "brightness" in classes
    assert "final_frame" in classes
