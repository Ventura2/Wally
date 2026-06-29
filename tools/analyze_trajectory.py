"""Analyze a wally agent trajectory npz for task-completion evidence.

Usage:
    python tools/analyze_trajectory.py <path/to/episode_0.npz> [--goal-frame PATH]

Looks at:
- events[*] for inventory changes, pickups, mine_block, break_item
- actions[*] for movement patterns (forward, attack, inventory spam)
- costs[*]  for L0 progress (per-step broadcast of per-replan plan_result.cost)
- frames vs. goal frame for visual similarity (best step, trend)
- frames metadata (size, count, brightness changes)

Verdict section at the bottom tells you whether the agent did anything useful.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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

DEFAULT_NPZ = Path(
    r"D:\Projects\Personal\artificial-intelligence\wally\ag-tests\run_wood\episode_0.npz"
)


def cost_progression_stats(costs: np.ndarray) -> dict[str, float]:
    """Summarize a per-step cost array into L0 progress metrics.

    `costs` is a length-T array where costs[t] is the most recent replan cost
    (broadcast across the replan window). Lower is closer to the goal.
    """
    n = len(costs)
    start = float(costs[0])
    end = float(costs[-1])
    min_v = float(costs.min())
    max_v = float(costs.max())
    min_idx = int(costs.argmin())
    max_idx = int(costs.argmax())
    mean_v = float(costs.mean())
    reduction_pct = (start - end) / abs(start) * 100.0 if start != 0 else 0.0
    below_half_pct = (
        float((costs < 0.5 * abs(start)).sum()) / n * 100.0 if start != 0 else 0.0
    )
    below_quarter_pct = (
        float((costs < 0.25 * abs(start)).sum()) / n * 100.0 if start != 0 else 0.0
    )
    trend_corr = 0.0
    if n > 1 and costs.std() > 0:
        steps = np.arange(n, dtype=np.float64)
        trend_corr = float(np.corrcoef(steps, costs)[0, 1])
    return {
        "n": n,
        "start": start,
        "end": end,
        "min": min_v,
        "max": max_v,
        "min_idx": min_idx,
        "max_idx": max_idx,
        "mean": mean_v,
        "reduction_pct": reduction_pct,
        "below_half_pct": below_half_pct,
        "below_quarter_pct": below_quarter_pct,
        "trend_corr": trend_corr,
    }


def goal_frame_similarity(
    frames: np.ndarray, goal_img_resized: np.ndarray
) -> dict[str, float]:
    """Compute per-frame MSE and a 0-1 similarity score against a goal image.

    `frames` is (T, H, W, 3) uint8. `goal_img_resized` is (H, W, 3) uint8 at the
    same H, W. Returns dict with per-frame arrays plus summary stats.
    """
    frames_f = frames.astype(np.float32)
    goal_f = goal_img_resized.astype(np.float32)
    mse_per_frame = ((frames_f - goal_f[None]) ** 2).mean(axis=(1, 2, 3))
    similarity = 1.0 / (1.0 + mse_per_frame / 1000.0)
    n = len(similarity)
    min_idx = int(similarity.argmax())
    trend_corr = 0.0
    if n > 1 and similarity.std() > 0:
        steps = np.arange(n, dtype=np.float64)
        trend_corr = float(np.corrcoef(steps, similarity)[0, 1])
    return {
        "mse_per_frame": mse_per_frame,
        "similarity": similarity,
        "n": n,
        "final_mse": float(mse_per_frame[-1]),
        "min_mse": float(mse_per_frame.min()),
        "final_sim": float(similarity[-1]),
        "max_sim": float(similarity.max()),
        "max_sim_idx": min_idx,
        "mean_sim": float(similarity.mean()),
        "trend_corr": trend_corr,
    }


def camera_shake_metrics(signal: np.ndarray, active_threshold: float = 0.01) -> dict[str, float]:
    """Quantify how 'shake-like' a per-step camera action signal is.

    The input is the per-step camera action (positive = one direction, negative =
    the other, near-zero = no rotation this step). Pure shake = lots of direction
    reversals that roughly cancel out; pure drift = sustained motion in one
    direction.

    Metrics (computed on "active" steps where |signal| > active_threshold):
    - n_active:      number of steps the agent actually rotated the camera
    - total_motion:  sum of |signal|          (total camera work)
    - net_motion:    |sum(signal)|            (signed displacement, ignores back-and-forth)
    - inefficiency:  net_motion / total_motion  (0 = pure shake, 1 = pure drift)
    - flip_rate:     fraction of consecutive active deltas with opposite sign
                                              (0 = monotonic, 1 = alternating)
    - var_active:    variance of the active signal values
    """
    abs_s = np.abs(signal)
    active = abs_s > active_threshold
    n_active = int(active.sum())
    total_motion = float(abs_s.sum())
    net_motion = float(abs(signal.sum()))
    inefficiency = net_motion / total_motion if total_motion > 0 else 0.0
    active_s = signal[active]
    signs = np.sign(active_s)
    if len(signs) > 1:
        flip_rate = float((signs[1:] * signs[:-1] < 0).sum()) / float(len(signs) - 1)
    else:
        flip_rate = 0.0
    var_active = float(active_s.var()) if n_active > 0 else 0.0
    return {
        "n_active": n_active,
        "total_motion": total_motion,
        "net_motion": net_motion,
        "inefficiency": inefficiency,
        "flip_rate": flip_rate,
        "var_active": var_active,
    }


def _shake_label(m: dict[str, float]) -> str:
    """Classify a per-axis shake metrics dict into a human label."""
    if m["flip_rate"] < 0.2:
        return "drift / purposeful"
    if m["flip_rate"] < 0.4:
        return "moderate shake"
    if m["inefficiency"] < 0.3:
        return "STRONG shake (lots of back-and-forth, low net motion)"
    return "mixed (shake + drift)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a wally agent trajectory npz"
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        type=Path,
        default=DEFAULT_NPZ,
        help="Path to episode_0.npz",
    )
    parser.add_argument(
        "--goal-frame",
        type=Path,
        default=None,
        help="Path to goal PNG for similarity analysis "
        "(default: <repo>/checkpoints/goal_frame1.png)",
    )
    args = parser.parse_args()
    npz_path = args.npz_path

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

    print("=== Cost progression (per-step, broadcast from per-replan plan_result.cost) ===")
    print("  Lower = closer to goal in latent space (the planner's own optimization target).")
    if "costs" in data.files:
        costs = data["costs"]
        print(f"  shape={costs.shape} dtype={costs.dtype}")
        s = cost_progression_stats(costs)
        print(f"  cost[start]   = {s['start']:>9.4f}")
        print(f"  cost[end]     = {s['end']:>9.4f}")
        print(f"  cost[min]     = {s['min']:>9.4f}  (at step {s['min_idx']})")
        print(f"  cost[max]     = {s['max']:>9.4f}  (at step {s['max_idx']})")
        print(f"  cost[mean]    = {s['mean']:>9.4f}")
        print(f"  reduction     = {s['reduction_pct']:>+8.2f}%  (start -> end)")
        print(f"  below 50% start:  {s['below_half_pct']:>5.1f}% of steps")
        print(f"  below 25% start:  {s['below_quarter_pct']:>5.1f}% of steps")
        print(
            f"  trend (corr cost vs step): {s['trend_corr']:+.3f}  "
            f"({'cost DROPPING over time' if s['trend_corr'] < -0.2 else 'cost RISING / flat'})"
        )
        if s["reduction_pct"] > 20 and s["trend_corr"] < -0.2:
            print("  >> L0 is making clear progress toward the goal in latent space")
        elif s["reduction_pct"] < -20 or s["trend_corr"] > 0.2:
            print("  >> L0 cost INCREASED over the episode — agent moved away from goal")
        else:
            print("  >> L0 cost is roughly flat — agent is not converging on the goal")
    else:
        print("  (not recorded in this npz — re-run with updated recorder to populate 'costs')")
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
    print("=== Camera shake analysis (cols 10=pitch, 11=yaw) ===")
    print("  Per-step actions are velocities: sign = direction, |val| = how much.")
    print("  High flip_rate + low inefficiency = shaking randomly in that axis.")
    pitch_m = camera_shake_metrics(actions[:, 10].astype(np.float64))
    yaw_m = camera_shake_metrics(actions[:, 11].astype(np.float64))
    header = (
        f"  {'axis':>5}  {'n_active':>8}  {'total_motion':>12}  "
        f"{'net_motion':>10}  {'ineff':>5}  {'flip_rate':>9}  {'var_active':>10}"
    )
    print(header)
    for name, m in (("pitch", pitch_m), ("yaw", yaw_m)):
        print(
            f"  {name:>5}  {m['n_active']:>8}  {m['total_motion']:>12.2f}  "
            f"{m['net_motion']:>10.2f}  {m['inefficiency']:>5.2f}  "
            f"{m['flip_rate']:>9.2f}  {m['var_active']:>10.3f}"
        )

    print()
    print(f"  pitch: {_shake_label(pitch_m)}")
    print(f"  yaw:   {_shake_label(yaw_m)}")

    both_strong = (
        pitch_m["flip_rate"] > 0.4
        and yaw_m["flip_rate"] > 0.4
        and pitch_m["inefficiency"] < 0.3
        and yaw_m["inefficiency"] < 0.3
    )
    both_moderate = (
        pitch_m["flip_rate"] > 0.3 and yaw_m["flip_rate"] > 0.3
    )
    print()
    if both_strong:
        print("  >> Agent appears to be rotating the camera RANDOMLY IN ALL DIRECTIONS")
        print("     (high sign-flip rate + low net motion on BOTH pitch and yaw)")
    elif both_moderate:
        print("  >> Agent shows moderate shake on BOTH axes (random-ish camera motion)")
    else:
        print("  >> No strong evidence of random shaking in all directions")

    print()
    print("=== Goal-frame similarity (pixel-space distance to --goal-frame) ===")
    print("  Measures how visually close the agent's view got to the goal image.")
    print("  Note: this is a *pixel* proxy, not the planner's latent cost.")
    goal_path = args.goal_frame
    if goal_path is None:
        goal_path = Path(__file__).resolve().parent.parent / "checkpoints" / "goal_frame1.png"
    if not goal_path.exists():
        print(f"  (goal frame not found at {goal_path} — pass --goal-frame PATH to enable)")
    else:
        h, w = frames.shape[1], frames.shape[2]
        goal_img = np.asarray(
            Image.open(goal_path).convert("RGB").resize((w, h)), dtype=np.uint8
        )
        sim = goal_frame_similarity(frames, goal_img)
        print(f"  goal:        {goal_path}")
        print(f"  frame size:  {h}x{w}  (goal resized to match)")
        print()
        print(f"  MSE          final={sim['final_mse']:>7.1f}  "
              f"min={sim['min_mse']:>7.1f}  mean={sim['mse_per_frame'].mean():>7.1f}")
        print(f"  similarity   final={sim['final_sim']:.3f}  "
              f"max={sim['max_sim']:.3f} (step {sim['max_sim_idx']})  "
              f"mean={sim['mean_sim']:.3f}")
        print(f"                (0=different, 1=identical — 1/(1+mse/1000) scale)")
        if sim["max_sim"] > sim["final_sim"]:
            gap = sim["max_sim"] - sim["final_sim"]
            tag = "got close then lost it" if gap > 0.05 else "kept best (or close)"
            print(f"  best - final: {gap:+.3f}  ({tag})")
        print(
            f"  trend (corr similarity vs step): {sim['trend_corr']:+.3f}  "
            f"({'trending TOWARD goal' if sim['trend_corr'] > 0.2 else 'NOT trending toward goal'})"
        )
        if sim["max_sim"] >= 0.95 and sim["final_sim"] >= 0.9:
            print("  >> Agent got visually close to the goal image")
        elif sim["max_sim"] >= 0.9 and sim["final_sim"] < 0.85:
            print("  >> Agent reached a goal-like view briefly, but did not stay there")
        elif sim["trend_corr"] > 0.3:
            print("  >> Agent is gradually approaching a goal-like view (slow but positive)")
        else:
            print("  >> Agent never produced a goal-like view in this episode")
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
    print("=== Same-candidate selection audit (SCSA, TRM paper App. B.1) ===")
    print("  Per replan: rank the final CEM population by the planner's cost")
    print("  and by the L2 diagnostic `||z_H - z_g||^2`, then compare.")
    print("  High Spearman + low oracle-rank = planner's cost picks good candidates.")
    scsa_keys = ("scsa_costs", "scsa_l2_costs")
    if all(k in data.files for k in scsa_keys):
        scsa_costs = data["scsa_costs"]
        scsa_l2 = data["scsa_l2_costs"]
        n_replans = scsa_costs.shape[0]
        pop = scsa_costs.shape[1]
        spearman_per = np.empty(n_replans, dtype=np.float64)
        oracle_rank_per = np.empty(n_replans, dtype=np.float64)
        for r in range(n_replans):
            p = scsa_costs[r]
            l = scsa_l2[r]
            if np.std(p) < 1e-9 or np.std(l) < 1e-9:
                spearman_per[r] = 0.0
                oracle_rank_per[r] = 50.0
                continue
            sp = np.corrcoef(p, l)[0, 1]
            spearman_per[r] = sp if np.isfinite(sp) else 0.0
            l2_best = int(np.argmin(l))
            sorted_by_p = np.argsort(p)
            rank = int(np.where(sorted_by_p == l2_best)[0][0])
            oracle_rank_per[r] = rank / max(pop - 1, 1) * 100.0
        print(f"  n_replans: {n_replans}   population_size: {pop}")
        print(
            f"  Spearman (planner vs L2): mean={spearman_per.mean():+.3f}  "
            f"median={np.median(spearman_per):+.3f}  "
            f"frac>0.5={float((spearman_per > 0.5).mean()):.2f}"
        )
        print(
            f"  Oracle-best rank percentile: mean={oracle_rank_per.mean():.1f}  "
            f"median={np.median(oracle_rank_per):.1f}  "
            f"(0 = picked the L2-best, 100 = picked the L2-worst)"
        )
        if spearman_per.mean() > 0.5 and oracle_rank_per.mean() < 30.0:
            print("  >> Planner's cost ranks candidates well; TRM head is unlikely to help much.")
            print("     Bottleneck is dynamics, not cost.")
        elif spearman_per.mean() < 0.2:
            print("  >> Planner's cost ranks candidates near-randomly.")
            print("     This is the 'latent-proximity trap' (TRM paper).")
            print("     A learned reachability head or subspace surgery may help.")
        else:
            print("  >> Planner's cost ranks candidates partially; mixed signal.")
    else:
        print("  (not recorded in this npz — re-run with the updated recorder)")

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
