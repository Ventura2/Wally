"""Generate the SCSA Spearman-distribution figure for the paper."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def per_replan_spearman(path: str) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    scsa_costs = d["scsa_costs"]
    scsa_l2 = d["scsa_l2_costs"]
    n_replans = scsa_costs.shape[0]
    spearman = np.empty(n_replans, dtype=np.float64)
    for r in range(n_replans):
        p = scsa_costs[r]
        l = scsa_l2[r]
        if np.std(p) < 1e-9 or np.std(l) < 1e-9:
            spearman[r] = 0.0
            continue
        s = np.corrcoef(p, l)[0, 1]
        spearman[r] = s if np.isfinite(s) else 0.0
    return spearman


def main() -> None:
    cosine = per_replan_spearman(
        r"D:\Projects\Personal\artificial-intelligence\wally\ag-tests\run_wood_1k_scsa2\episode_0.npz"
    )
    cos_trm = per_replan_spearman(
        r"D:\Projects\Personal\artificial-intelligence\wally\ag-tests\run_wood_1k_trm\episode_0.npz"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(cosine, bins=30, color="tab:gray", edgecolor="black", alpha=0.85)
    axes[0].axvline(np.mean(cosine), color="red", linestyle="--", label=f"mean={np.mean(cosine):.3f}")
    axes[0].axvline(np.median(cosine), color="blue", linestyle=":", label=f"median={np.median(cosine):.3f}")
    axes[0].set_xlabel("Spearman (planner cost vs L2 diagnostic), per replan")
    axes[0].set_ylabel("replans")
    axes[0].set_title("Cosine cost only")
    axes[0].set_xlim(-0.5, 1.0)
    axes[0].legend()

    axes[1].hist(cos_trm, bins=30, color="tab:green", edgecolor="black", alpha=0.85)
    axes[1].axvline(np.mean(cos_trm), color="red", linestyle="--", label=f"mean={np.mean(cos_trm):.3f}")
    axes[1].axvline(np.median(cos_trm), color="blue", linestyle=":", label=f"median={np.median(cos_trm):.3f}")
    axes[1].set_xlabel("Spearman (planner cost vs L2 diagnostic), per replan")
    axes[1].set_ylabel("replans")
    axes[1].set_title("Cosine + TRM head")
    axes[1].set_xlim(-0.5, 1.0)
    axes[1].legend()

    fig.suptitle("Same-Candidate Selection Audit (SCSA): per-replan ranking agreement")
    fig.tight_layout()
    out = r"D:\Projects\Personal\artificial-intelligence\wally\docs\papers\fig_scsa_spearman.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  cosine:  mean={cosine.mean():.3f}  median={np.median(cosine):.3f}  frac>0.5={(cosine > 0.5).mean():.2f}")
    print(f"  cos+trm: mean={cos_trm.mean():.3f}  median={np.median(cos_trm):.3f}  frac>0.5={(cos_trm > 0.5).mean():.2f}")


if __name__ == "__main__":
    main()
