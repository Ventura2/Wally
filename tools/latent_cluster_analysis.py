"""Latent-clustering analysis for the LeWorldModel encoder.

Question: do the LeWorldModel's CNN latents already cluster by Minecraft
milestone (log acquired, planks crafted, wooden pickaxe, iron pickaxe...)?
If yes, a vanilla latent-distance CEM is enough to plan at the milestone
level. If no, we need an explicit skip-k prediction head or shaped cost.

Inputs:
- LeWorldModel checkpoint (default: checkpoints/checkpoint_100000.pt)
- MineRL rendered.npz ZIP for milestone labels (default: IronPickaxe)
- raw .tar shards for the actual POV frames (default: data/raw/minerl_iron)

Output (in --output-dir):
- report.md + report.json
- latents.npz   (latents, milestone_ids, cluster_ids, episode_ids, step_indices)
- pca_clusters.png, pca_milestones.png, milestone_timeline.png
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import tarfile
import tempfile
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("latent_cluster_analysis")


MILESTONES: list[tuple[str, str]] = [
    # (name, npz_key)  — checked in order, max index where condition is true wins
    ("log_acquired", "observation$inventory$log"),
    ("planks_crafted", "observation$inventory$planks"),
    ("crafting_table_placed", "observation$inventory$crafting_table"),
    ("sticks_crafted", "observation$inventory$stick"),
    ("wooden_pickaxe", "observation$inventory$wooden_pickaxe"),
    ("stone_pickaxe", "observation$inventory$stone_pickaxe"),
    ("iron_ore_mined", "observation$inventory$iron_ore"),
    ("iron_ingot_smelted", "observation$inventory$iron_ingot"),
    ("iron_pickaxe_obtained", "observation$inventory$iron_pickaxe"),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_default_zip_name(zip_path: Path) -> str:
    return zip_path.stem


def _zip_traj_prefix(zip_name: str, traj_name: str) -> str:
    return f"{zip_name}/{traj_name}/"


def _raw_episode_id_to_traj(ep_id: str, zip_name: str) -> str:
    prefix = f"{zip_name}_"
    if ep_id.startswith(prefix):
        return ep_id[len(prefix):]
    return ep_id


def _compute_milestones(obs: np.ndarray) -> np.ndarray:
    """Return per-step milestone id (0 = none yet, 1..N = highest reached)."""
    n = obs.shape[0]
    ms = np.zeros(n, dtype=np.int64)
    for i, (_name, key) in enumerate(MILESTONES, start=1):
        if key not in obs.dtype.names and key not in obs:
            # dict-of-arrays format: keys live in obs directly
            pass
        v = obs[key]
        reached = (v >= 1).astype(np.int64)
        ms = np.where(reached & (np.arange(n) >= 0), np.maximum(ms, i * reached), ms)
    return ms


def _compute_milestones_from_npz(npz_path: str) -> np.ndarray:
    """Compute milestone ids for all frames in a MineRL rendered.npz file.

    rendered.npz stores observation$* (T+1 frames) and action$* (T actions).
    We use observation[t] for t in [0..T], matching the raw shard jpg key
    <ep_id>_<t:06d>.jpg convention (frame t is the state when action t-1
    was just taken, or the initial state for t=0).
    """
    d = np.load(npz_path)
    n = None
    for _name, key in MILESTONES:
        if key in d:
            arr = d[key]
            n = arr.shape[0] if n is None else n
            break
    if n is None:
        raise RuntimeError(f"No milestone keys found in {npz_path}")
    ms = np.zeros(n, dtype=np.int64)
    for i, (_name, key) in enumerate(MILESTONES, start=1):
        if key not in d:
            continue
        v = np.asarray(d[key])
        ms = np.maximum(ms, i * (v >= 1).astype(np.int64))
    return ms


def _iter_raw_shard_episodes(
    raw_dir: Path,
    ep_filter: set[str] | None = None,
) -> dict[str, list[int]]:
    """Return {ep_id: sorted [step_indices]} from raw .tar shards in raw_dir.

    Streams the tar entries (avoid ``getmembers()`` which on Windows forces
    the whole central directory into memory and is the slowest path).
    If ``ep_filter`` is given, only JSONs whose ep_id is in the set are read.
    """
    eps: dict[str, list[int]] = {}
    for shard in sorted(raw_dir.glob("shard_*.tar")):
        with tarfile.open(shard, "r") as t:
            for m in t:
                if not m.isfile() or not m.name.endswith(".json"):
                    continue
                # ep_id = everything before the last _<6 digits>
                base = m.name[:-5]  # strip .json
                ep, step_str = base.rsplit("_", 1)
                if not step_str.isdigit():
                    continue
                if ep_filter is not None and ep not in ep_filter:
                    continue
                eps.setdefault(ep, []).append(int(step_str))
    for ep in eps:
        eps[ep].sort()
    return eps


def _load_jpg_bytes_parallel(
    raw_dir: Path,
    requests: list[tuple[str, int]],
    max_workers: int = 8,
) -> list[bytes]:
    """Read many (ep_id, step) jpgs. Returns bytes in the same order.

    Optimization: open each .tar shard at most once and extract every
    matching member in a single streaming pass. The raw shards are
    written so episodes are never split across shards, so for each
    requested (ep_id, step) we only need to scan one shard.
    """
    by_target: dict[str, int] = {f"{ep}_{step:06d}.jpg": i for i, (ep, step) in enumerate(requests)}
    out: list[bytes | None] = [None] * len(requests)
    shards = sorted(raw_dir.glob("shard_*.tar"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.tar files in {raw_dir}")

    def _scan_shard(shard: Path) -> int:
        found = 0
        with tarfile.open(shard, "r") as t:
            for m in t:
                if not m.isfile() or not m.name.endswith(".jpg"):
                    continue
                idx = by_target.get(m.name)
                if idx is None:
                    continue
                f = t.extractfile(m)
                if f is not None:
                    out[idx] = f.read()
                    found += 1
        return found

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for n_found in pool.map(_scan_shard, shards):
            if n_found == 0:
                continue

    missing = [r for r, b in zip(requests, out) if b is None]
    if missing:
        raise FileNotFoundError(f"{len(missing)} jpgs not found, e.g. {missing[0]}")
    return [b if b is not None else b"" for b in out]  # type: ignore[return-value]


def _load_npz_for_episode(zip_path: Path, zip_name: str, traj_name: str) -> np.ndarray:
    """Extract rendered.npz for a traj into a temp file and return milestone ids."""
    prefix = _zip_traj_prefix(zip_name, traj_name)
    target = None
    with zipfile.ZipFile(zip_path, "r") as zf:
        for n in zf.namelist():
            if n.startswith(prefix) and n.endswith("rendered.npz"):
                target = n
                break
        if target is None:
            raise FileNotFoundError(f"rendered.npz not found for traj {traj_name}")
        with tempfile.TemporaryDirectory() as tmp:
            zf.extract(target, tmp)
            return _compute_milestones_from_npz(os.path.join(tmp, target))


class SimpleCNNEncoder(nn.Module):
    """Identical to src/wally/models/cnn_encoder.py:SimpleCNNEncoder.

    Reimplemented here (rather than imported) so the script stays a single
    file that can run against an arbitrary checkpoint without depending
    on the wally package import path.
    """

    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, embed_dim, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        return x.mean(dim=[2, 3])


def _load_encoder(checkpoint_path: Path, device: torch.device) -> tuple[SimpleCNNEncoder, int]:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    encoder = SimpleCNNEncoder()
    encoder_sd = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing:
        logger.warning("Encoder missing keys: %s", missing[:5])
    if unexpected:
        logger.warning("Encoder unexpected keys: %s", unexpected[:5])
    encoder.eval()
    encoder.to(device)
    return encoder, int(ck.get("global_step", -1))


def _preprocess_frames(jpg_bytes_list: list[bytes], device: torch.device) -> torch.Tensor:
    """Decode JPEGs to (N, 3, 224, 224) float tensor in [0, 1]."""
    arrs: list[np.ndarray] = []
    for b in jpg_bytes_list:
        img = Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224), Image.BILINEAR)
        arrs.append(np.asarray(img, dtype=np.uint8))
    a = np.stack(arrs, axis=0)
    t = torch.from_numpy(a).to(device)
    t = t.permute(0, 3, 1, 2).contiguous().float() / 255.0
    return t


@torch.no_grad()
def _encode_batches(
    encoder: SimpleCNNEncoder,
    jpg_bytes_list: list[bytes],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for i in range(0, len(jpg_bytes_list), batch_size):
        chunk = jpg_bytes_list[i : i + batch_size]
        x = _preprocess_frames(chunk, device)
        z = encoder(x)
        out.append(z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def _kmeans_np(x: np.ndarray, k: int, seed: int = 0, max_iter: int = 100, n_init: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy k-means with multiple restarts. Returns (assignments, centers)."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    best_inertia = np.inf
    best_a: np.ndarray | None = None
    best_c: np.ndarray | None = None
    for restart in range(n_init):
        idx = rng.choice(n, size=k, replace=False)
        centers = x[idx].copy()
        for _it in range(max_iter):
            dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
            a = np.argmin(dists, axis=1)
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = a == j
                if mask.any():
                    new_centers[j] = x[mask].mean(axis=0)
                else:
                    new_centers[j] = x[rng.integers(0, n)]
            shift = np.linalg.norm(new_centers - centers, axis=1).max()
            centers = new_centers
            if shift < 1e-4:
                break
        dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        a = np.argmin(dists, axis=1)
        inertia = float(((x - centers[a]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_a = a
            best_c = centers
    assert best_a is not None and best_c is not None
    return best_a, best_c


def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Contingency matrix n_a x n_b. a, b are integer label arrays."""
    a_unique = np.unique(a)
    b_unique = np.unique(b)
    ia = {v: i for i, v in enumerate(a_unique)}
    ib = {v: i for i, v in enumerate(b_unique)}
    M = np.zeros((len(a_unique), len(b_unique)), dtype=np.int64)
    for x, y in zip(a, b):
        M[ia[x], ib[y]] += 1
    return M


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized mutual information (symmetric, arithmetic mean of entropies)."""
    M = _contingency(a, b).astype(np.float64)
    n = M.sum()
    if n == 0:
        return 0.0
    pa = M.sum(axis=1, keepdims=True) / n
    pb = M.sum(axis=0, keepdims=True) / n
    pab = M / n
    with np.errstate(divide="ignore", invalid="ignore"):
        joint_ratio = pab / (pa * pb)
    nz = pab > 0
    mi = float((pab[nz] * np.log(joint_ratio[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    if ha == 0 or hb == 0:
        return 0.0
    return 2.0 * mi / (ha + hb)


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index."""
    M = _contingency(a, b).astype(np.float64)
    n = M.sum()
    if n < 2:
        return 0.0
    sum_comb_c = float((M.sum(axis=1) * (M.sum(axis=1) - 1) / 2).sum())
    sum_comb_k = float((M.sum(axis=0) * (M.sum(axis=0) - 1) / 2).sum())
    sum_comb = float((M * (M - 1) / 2).sum())
    expected = sum_comb_c * sum_comb_k / (n * (n - 1) / 2) if n > 1 else 0.0
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    if max_index == expected:
        return 1.0
    return (sum_comb - expected) / (max_index - expected)


def _homogeneity_completeness(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Homogeneity and completeness (Rosenberg & Hirschberg, 2007).

    a = cluster labels, b = ground-truth class labels.

    homogeneity = 1 - H(C|K)/H(C)  (each cluster contains only one class)
    completeness = 1 - H(K|C)/H(K) (all members of a class are in one cluster)
    """
    M = _contingency(a, b).astype(np.float64)
    n = M.sum()
    if n == 0:
        return 0.0, 0.0
    pab = M / n
    p_c = pab.sum(axis=1)  # (n_clusters,)
    p_k = pab.sum(axis=0)  # (n_classes,)
    h_c = float(-(p_c[p_c > 0] * np.log(p_c[p_c > 0])).sum())
    h_k = float(-(p_k[p_k > 0] * np.log(p_k[p_k > 0])).sum())
    # H(C|K) = sum_k p_k * H(C|k), p(c|k) = p_ck / p_k
    p_ck = pab / np.maximum(p_k[None, :], 1e-30)
    p_ck = np.where(p_ck > 0, p_ck, 1.0)  # avoid log(0); rows with all zero contribute 0
    h_c_given_k_per_k = -np.sum(p_ck * np.log(p_ck), axis=0)
    h_c_given_k = float(np.sum(p_k * h_c_given_k_per_k))
    # H(K|C) = sum_c p_c * H(K|c), p(k|c) = p_ck / p_c
    p_kc = pab / np.maximum(p_c[:, None], 1e-30)
    p_kc = np.where(p_kc > 0, p_kc, 1.0)
    h_k_given_c_per_c = -np.sum(p_kc * np.log(p_kc), axis=1)
    h_k_given_c = float(np.sum(p_c * h_k_given_c_per_c))
    hom = max(0.0, 1.0 - h_c_given_k / h_c) if h_c > 0 else 0.0
    com = max(0.0, 1.0 - h_k_given_c / h_k) if h_k > 0 else 0.0
    return float(hom), float(com)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def _scatter_plot(
    xy: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str = "tab20",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap=cmap, s=4, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="label")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _milestone_timeline_plot(
    milestones: np.ndarray,
    out_path: Path,
    max_examples: int = 6,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 0.55 * max_examples + 1))
    n_total = len(milestones)
    n_per = n_total // max_examples if n_total >= max_examples else 1
    row_offset = len(MILESTONES) + 1
    for row in range(max_examples):
        start = row * n_per
        end = (row + 1) * n_per if row < max_examples - 1 else n_total
        if start >= n_total:
            break
        ms = milestones[start:end]
        ax.plot(
            np.arange(start, end),
            ms + row * row_offset,
            drawstyle="steps-post",
            color="tab:blue",
            linewidth=1.4,
        )
    ax.set_yticks([])
    ax.set_xlabel("global frame index")
    ax.set_title("Milestone timeline (each row = one episode, stacked)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_repo_root() / "checkpoints" / "checkpoint_100000.pt")
    parser.add_argument("--zip", type=Path, default=_repo_root() / "data" / "external" / "MineRLObtainIronPickaxe-v0.zip")
    parser.add_argument("--raw-shards-dir", type=Path, default=_repo_root() / "data" / "raw" / "minerl_iron")
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--subsample-stride", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=_repo_root() / "tools" / "latent_analysis_output")
    parser.add_argument("--decode-workers", type=int, default=8, help="Thread-pool size for parallel JPEG decode from raw shards")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Re-run the analysis (k-means, scores, plots, report) from the latents.npz in --output-dir instead of re-encoding.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.zip.exists():
        raise SystemExit(f"zip not found: {args.zip}")
    if not args.raw_shards_dir.exists():
        raise SystemExit(f"raw shards dir not found: {args.raw_shards_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_name = _resolve_default_zip_name(args.zip)
    device = torch.device(args.device)

    t0 = time.time()
    all_latents: list[np.ndarray] = []
    all_milestones: list[np.ndarray] = []
    all_ep_ids: list[str] = []
    all_step_indices: list[np.ndarray] = []

    if args.from_cache:
        cache = args.output_dir / "latents.npz"
        if not cache.exists():
            raise SystemExit(f"--from-cache: {cache} not found")
        logger.info("Loading cached latents from %s", cache)
        d = np.load(cache, allow_pickle=True)
        Z = d["latents"].astype(np.float32)
        M = d["milestones"].astype(np.int64)
        E = d["episode_ids"]
        S = d["step_indices"].astype(np.int64)
        # Group by episode to reconstruct the per-episode lists
        unique_eps = list(dict.fromkeys(E.tolist()))
        for ep in unique_eps:
            mask = E == ep
            all_latents.append(Z[mask])
            all_milestones.append(M[mask])
            all_ep_ids.append(str(ep))
            all_step_indices.append(S[mask])
        # Restack
        Z = np.concatenate(all_latents, axis=0)
        M = np.concatenate(all_milestones, axis=0)
        E = np.array(sum(([ep] * len(z) for ep, z in zip(all_ep_ids, all_latents)), []), dtype=object)
        S = np.concatenate(all_step_indices, axis=0)
        global_step = -1
    else:
        logger.info("Loading encoder from %s", args.checkpoint)
        encoder, global_step = _load_encoder(args.checkpoint, device)
        logger.info("Encoder loaded (global_step=%d), loading on %s", global_step, device)

        # Restrict raw-shard scan to episodes that exist in the ZIP.
        logger.info("Listing ZIP namelist to build ep_filter")
        with zipfile.ZipFile(args.zip, "r") as zf:
            zip_traj_names = {
                n.strip("/").split("/")[1]
                for n in zf.namelist()
                if n.startswith(f"{zip_name}/") and len(n.strip("/").split("/")) >= 2
            }
        ep_filter = {f"{zip_name}_{t}" for t in zip_traj_names}
        logger.info("ZIP has %d trajs, ep_filter has %d ep_ids", len(zip_traj_names), len(ep_filter))

        logger.info("Indexing raw shard episodes in %s (filtered)", args.raw_shards_dir)
        raw_eps = _iter_raw_shard_episodes(args.raw_shards_dir, ep_filter=ep_filter)
        if not raw_eps:
            raise SystemExit(f"no episodes found in {args.raw_shards_dir}")
        raw_ep_ids = sorted(raw_eps.keys())[: args.max_episodes]
        logger.info("Selected %d episodes: %s ...", len(raw_ep_ids), raw_ep_ids[:2])

        for ep_id in raw_ep_ids:
            traj_name = _raw_episode_id_to_traj(ep_id, zip_name)
            try:
                ms = _load_npz_for_episode(args.zip, zip_name, traj_name)
            except FileNotFoundError as e:
                logger.warning("Skipping %s: %s", ep_id, e)
                continue
            steps_all = raw_eps[ep_id]
            steps = steps_all[:: args.subsample_stride]
            steps = [s for s in steps if s < len(ms)]
            if not steps:
                logger.warning("Skipping %s: no subsampled steps fit", ep_id)
                continue
            t_dl = time.time()
            jpg_bytes = _load_jpg_bytes_parallel(
                args.raw_shards_dir,
                [(ep_id, s) for s in steps],
                max_workers=args.decode_workers,
            )
            t_decode = time.time() - t_dl
            t_enc = time.time()
            z = _encode_batches(encoder, jpg_bytes, device, args.batch_size)
            t_encode = time.time() - t_enc
            m = np.array([ms[s] for s in steps], dtype=np.int64)
            all_latents.append(z)
            all_milestones.append(m)
            all_ep_ids.append(ep_id)
            all_step_indices.append(np.array(steps, dtype=np.int64))
            logger.info(
                "Encoded %s: %d frames (decode %.1fs, encode %.1fs)",
                ep_id, len(jpg_bytes), t_decode, t_encode,
            )

        if not all_latents:
            raise SystemExit("no latents encoded — aborting")

        Z = np.concatenate(all_latents, axis=0)
        M = np.concatenate(all_milestones, axis=0)
        E = np.array(sum(([ep] * len(z) for ep, z in zip(all_ep_ids, all_latents)), []), dtype=object)
        S = np.concatenate(all_step_indices, axis=0)
        logger.info("Total: %d frames, latent dim %d", Z.shape[0], Z.shape[1])

    Z = np.concatenate(all_latents, axis=0)
    M = np.concatenate(all_milestones, axis=0)
    E = np.array(sum(([ep] * len(z) for ep, z in zip(all_ep_ids, all_latents)), []), dtype=object)
    S = np.concatenate(all_step_indices, axis=0)
    logger.info("Total: %d frames, latent dim %d", Z.shape[0], Z.shape[1])

    logger.info("Running k-means k=%d", args.n_clusters)
    clusters, centers = _kmeans_np(Z, args.n_clusters, seed=args.seed, n_init=4, max_iter=100)
    inertia = float(((Z - centers[clusters]) ** 2).sum())
    logger.info("k-means done, inertia=%.2f", inertia)

    nmi = _nmi(clusters, M)
    ari = _ari(clusters, M)
    hom, com = _homogeneity_completeness(clusters, M)
    logger.info("NMI=%.3f ARI=%.3f hom=%.3f com=%.3f", nmi, ari, hom, com)

    # Random baseline: shuffle cluster assignments and rescore
    rng = np.random.default_rng(args.seed)
    rand = rng.permutation(clusters)
    nmi_rnd = _nmi(rand, M)
    ari_rnd = _ari(rand, M)
    hom_rnd, com_rnd = _homogeneity_completeness(rand, M)

    # Per-milestone purity: of frames at milestone k, fraction in largest cluster
    purity_per_milestone: list[dict[str, Any]] = []
    for k in range(0, len(MILESTONES) + 1):
        mask = M == k
        if mask.sum() == 0:
            continue
        c = Counter(clusters[mask].tolist())
        top = c.most_common(1)[0]
        purity_per_milestone.append({
            "milestone_index": k,
            "milestone_name": "none" if k == 0 else MILESTONES[k - 1][0],
            "n_frames": int(mask.sum()),
            "top_cluster": int(top[0]),
            "top_cluster_share": float(top[1] / mask.sum()),
        })

    # 2D PCA + scatter
    logger.info("Computing PCA-2D")
    xy = _pca_2d(Z)
    _scatter_plot(xy, clusters, f"k-means clusters (k={args.n_clusters})", args.output_dir / "pca_clusters.png")
    _scatter_plot(xy, M, "Milestone labels (ground truth from inventory)", args.output_dir / "pca_milestones.png")
    # Stacked milestone timeline
    if all_milestones:
        all_ms_concat = np.concatenate(all_milestones, axis=0)
        _milestone_timeline_plot(all_ms_concat, args.output_dir / "milestone_timeline.png")

    # Per-milestone between-centroid distance vs within-milestone distance
    # (latent-space "linear separability" probe)
    between_dist = []
    within_dist = []
    for k in range(0, len(MILESTONES) + 1):
        mask = M == k
        if mask.sum() < 2:
            continue
        centroid = Z[mask].mean(axis=0)
        within_dist.append(float(np.linalg.norm(Z[mask] - centroid, axis=1).mean()))
        between_dist.append(float(np.linalg.norm(centroid - Z.mean(axis=0))))
    between_mean = float(np.mean(between_dist)) if between_dist else 0.0
    within_mean = float(np.mean(within_dist)) if within_dist else 0.0
    separation_ratio = between_mean / within_mean if within_mean > 0 else 0.0

    # Linear progress probe: regress milestone index on the first K PCA components
    # using a closed-form ridge. High R^2 means the latent is linearly ordered
    # by progress even when k-means can't separate the milestones.
    pca_components = _pca_2d(Z)
    pca_full = pca_components  # already the top-2 components
    # Add 3rd and 4th components for the regression
    Zc = Z - Z.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(Zc, full_matrices=False)
    pca_top = Zc @ vt[:4].T
    X = np.hstack([pca_top, np.ones((len(M), 1))])
    y = M.astype(np.float64)
    lam = 1e-3
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    y_pred = X @ w
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    progress_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Also report PC1-only R^2 (the gradient visible in the plot)
    X1 = np.hstack([pca_top[:, :1], np.ones((len(M), 1))])
    w1 = np.linalg.solve(X1.T @ X1 + lam * np.eye(2), X1.T @ y)
    y_pred1 = X1 @ w1
    pc1_r2 = 1.0 - float(((y - y_pred1) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    logger.info("linear progress R^2 (PC1..4): %.3f | PC1-only R^2: %.3f", progress_r2, pc1_r2)

    elapsed = time.time() - t0
    report = {
        "checkpoint": str(args.checkpoint),
        "global_step": global_step,
        "zip": str(args.zip),
        "raw_shards_dir": str(args.raw_shards_dir),
        "n_episodes": len(all_ep_ids),
        "n_frames": int(Z.shape[0]),
        "latent_dim": int(Z.shape[1]),
        "subsample_stride": args.subsample_stride,
        "n_clusters": args.n_clusters,
        "kmeans_inertia": inertia,
        "scores": {
            "nmi": nmi,
            "ari": ari,
            "homogeneity": hom,
            "completeness": com,
            "nmi_random_baseline": nmi_rnd,
            "ari_random_baseline": ari_rnd,
            "homogeneity_random_baseline": hom_rnd,
            "completeness_random_baseline": com_rnd,
        },
        "latent_separation": {
            "between_milestone_centroids_mean": between_mean,
            "within_milestone_spread_mean": within_mean,
            "ratio_between_over_within": separation_ratio,
            "linear_progress_r2_pc1_to_pc4": progress_r2,
            "pc1_only_r2": pc1_r2,
        },
        "milestone_purity": purity_per_milestone,
        "milestone_names": [m[0] for m in MILESTONES],
        "episodes": all_ep_ids,
        "elapsed_seconds": elapsed,
    }

    np.savez_compressed(
        args.output_dir / "latents.npz",
        latents=Z,
        milestones=M,
        clusters=clusters,
        episode_ids=E,
        step_indices=S,
    )

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    md_lines: list[str] = []
    md_lines.append(f"# Latent-clustering analysis\n")
    md_lines.append(f"- checkpoint: `{args.checkpoint}` (step {global_step})")
    md_lines.append(f"- zip: `{args.zip}`")
    md_lines.append(f"- raw shards: `{args.raw_shards_dir}`")
    md_lines.append(f"- episodes: {len(all_ep_ids)}")
    md_lines.append(f"- frames (after subsample x{args.subsample_stride}): {Z.shape[0]}")
    md_lines.append(f"- latent dim: {Z.shape[1]}")
    md_lines.append(f"- k-means k: {args.n_clusters}  (inertia {inertia:.1f})")
    md_lines.append(f"- elapsed: {elapsed:.1f}s\n")
    md_lines.append("## Cluster vs milestone agreement\n")
    md_lines.append("| metric | observed | random baseline |")
    md_lines.append("|---|---:|---:|")
    md_lines.append(f"| NMI (geometric mean) | {nmi:.3f} | {nmi_rnd:.3f} |")
    md_lines.append(f"| ARI (adjusted Rand) | {ari:.3f} | {ari_rnd:.3f} |")
    md_lines.append(f"| Homogeneity | {hom:.3f} | {hom_rnd:.3f} |")
    md_lines.append(f"| Completeness | {com:.3f} | {com_rnd:.3f} |")
    md_lines.append("")
    md_lines.append("## Latent separation\n")
    md_lines.append(f"- between-milestone centroid distance: {between_mean:.3f}")
    md_lines.append(f"- within-milestone spread: {within_mean:.3f}")
    md_lines.append(f"- ratio (between / within): {separation_ratio:.3f}")
    md_lines.append(f"- linear progress R^2 (PC1..4 → milestone index): {progress_r2:.3f}")
    md_lines.append(f"- PC1-only R^2: {pc1_r2:.3f}\n")
    md_lines.append("## Per-milestone purity\n")
    md_lines.append("| milestone | n_frames | top cluster share |")
    md_lines.append("|---|---:|---:|")
    for p in purity_per_milestone:
        md_lines.append(f"| {p['milestone_name']} | {p['n_frames']} | {p['top_cluster_share']:.2f} |")
    md_lines.append("\n## Verdict\n")
    # Compute a noise floor: how well does random clustering agree with milestones?
    rnd_nmi = report["scores"]["nmi_random_baseline"]
    rnd_ari = report["scores"]["ari_random_baseline"]
    excess_nmi = nmi - rnd_nmi
    excess_ari = ari - rnd_ari
    md_lines.append(f"observed NMI = {nmi:.3f} (random baseline {rnd_nmi:.3f}, excess {excess_nmi:+.3f})")
    md_lines.append(f"observed ARI = {ari:.3f} (random baseline {rnd_ari:.3f}, excess {excess_ari:+.3f})")
    md_lines.append(f"frames = {Z.shape[0]}  episodes = {len(all_ep_ids)}\n")
    if Z.shape[0] < 200:
        md_lines.append(
            f"_Sample is too small ({Z.shape[0]} frames) to draw a confident verdict. "
            "Re-run with `--max-episodes 5 --subsample-stride 5` for the full analysis._"
        )
    elif excess_nmi > 0.3 and excess_ari > 0.2:
        md_lines.append(
            "**Latents already cluster strongly by milestone.** "
            "Skip-k head is **not needed** for milestone-level planning; "
            "CEM on existing latents should work. Proceed to milestone-mining + shaped-cost."
        )
    elif excess_nmi > 0.1:
        md_lines.append(
            "**Latents have partial milestone structure.** "
            "CEM on existing latents may work for short horizons, "
            "but the skip-k head is **likely needed** to amplify temporal abstraction for long-horizon tasks."
        )
    else:
        md_lines.append(
            "**Latents are mostly pixel-similar, not milestone-clustered.** "
            "Skip-k head (or other temporal-abstraction head) is **required** — the existing latents are too low-level to plan at the milestone level."
        )
    md_lines.append("\n## Plots\n")
    md_lines.append("- `pca_clusters.png` — PCA-2D colored by k-means cluster")
    md_lines.append("- `pca_milestones.png` — PCA-2D colored by milestone (ground truth)")
    md_lines.append("- `milestone_timeline.png` — stacked per-episode milestone progression\n")
    (args.output_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("Done. Report at %s/report.md", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
