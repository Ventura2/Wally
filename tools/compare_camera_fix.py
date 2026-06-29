"""Compare all wally agent runs side by side."""
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Projects\Personal\artificial-intelligence\wally")
RUNS = {
    "5k+L1v2 (HIER)": ROOT / "ag-tests" / "run_5k_l1_v2_demo" / "episode_0.npz",
    "5k+TRM (no hier)": ROOT / "ag-tests" / "run_wood_5k_trm_fixed" / "episode_0.npz",
    "5k (no hier) demo": ROOT / "ag-tests" / "run_wood_5k_trm_demo" / "episode_0.npz",
    "5k+TRM (broken)": ROOT / "ag-tests" / "run_wood_5k_trm" / "episode_0.npz",
}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a_r = np.argsort(np.argsort(a))
    b_r = np.argsort(np.argsort(b))
    a_r = a_r - a_r.mean()
    b_r = b_r - b_r.mean()
    denom = np.sqrt((a_r ** 2).sum() * (b_r ** 2).sum())
    return float((a_r * b_r).sum() / denom) if denom > 0 else 0.0


for label, path in RUNS.items():
    if not path.exists():
        print(f"=== {label} ===  (not found: {path.name})")
        continue
    d = np.load(path, allow_pickle=True)
    a = d["actions"]
    frames = d["frames"]
    bright = frames.mean(axis=(1, 2, 3))
    print(f"=== {label} ===")
    print(f"  steps: {a.shape[0]}")
    print(f"  camera_pitch (idx 0):  range=[{a[:,0].min():+.3f}, {a[:,0].max():+.3f}]  std={a[:,0].std():.4f}")
    print(f"  camera_yaw   (idx 1):  range=[{a[:,1].min():+.3f}, {a[:,1].max():+.3f}]  std={a[:,1].std():.4f}")
    print(f"  attack (idx 10) > 0.5: {(a[:,10] > 0.5).sum()} of {a.shape[0]} steps")
    print(f"  inventory (idx 12) max: {a[:,12].max():.3f}  (should be 0)")
    print(f"  brightness: start50={bright[:50].mean():.1f}  end50={bright[-50:].mean():.1f}  last5={[f'{b:.0f}' for b in bright[-5:]]}")
    if "costs" in d.files:
        c = d["costs"]
        red = (c[0] - c[-1]) / abs(c[0]) * 100 if c[0] != 0 else 0
        print(f"  cost: start={c[0]:+.3f}  end={c[-1]:+.3f}  reduction={red:+.1f}%")
    if "scsa_costs" in d.files and "scsa_l2_costs" in d.files:
        sc, sl = d["scsa_costs"], d["scsa_l2_costs"]
        rhos = [spearman(np.asarray(sc[i]).ravel(), np.asarray(sl[i]).ravel())
                for i in range(len(sc))
                if len(np.asarray(sc[i]).ravel()) >= 2]
        rhos = np.array(rhos)
        print(f"  SCSA Spearman: mean={rhos.mean():+.3f}  frac>0.5={float((rhos > 0.5).mean()):.0%}")
    print()
