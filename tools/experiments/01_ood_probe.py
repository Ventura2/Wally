"""01_ood_probe.py — Tier 1A + 1G.

Encodes synthetic "OOD" frames (sky, ground, water, dark cave, tree trunk
close, tree trunk far) through the L0 encoder and reports:
  (i)   per-frame latent L2 norms vs the training distribution's mean
  (ii)  pairwise L2 distances in latent space
  (iii) the L0 predictor's 1-step MSE when given a real frame + action
        vs a sky frame + action
  (iv) the predicted next-latent change for the action-magnitude
        variants (clipping the camera dim 10/11 to ±0.3 vs full ±1.0
        in the AGENT-vocab space, which the planner translates into
        different L0-space magnitudes).

Writes a summary table to stdout and saves a 2-D PCA / 2-D distances
plot to tools/experiments/_figures/01_ood_latents.png.
"""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("D:/Projects/Personal/artificial-intelligence/wally")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wally.planner.rollout import LatentRollout

CKPT = PROJECT_ROOT / "checkpoints" / "wood_1000" / "checkpoint_1000.pt"
SHARD = PROJECT_ROOT / "data" / "shards" / "treechop_full" / "shard_000001.tar"
FIG_DIR = PROJECT_ROOT / "tools" / "experiments" / "_figures"
FIG_DIR.mkdir(exist_ok=True)


def load_first_npz(tar_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".npz"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                buf = io.BytesIO(f.read())
                with np.load(buf) as data:
                    return data["frames"], data["actions"]
    raise RuntimeError(f"No .npz found in {tar_path}")


def synth_sky() -> np.ndarray:
    """Sky: top 70% light blue, bottom 30% white."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:45, :, :] = (135, 206, 235)   # sky blue
    img[45:, :, :] = (220, 220, 220)   # overcast horizon
    return img


def synth_ground() -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, :] = (60, 130, 60)      # grass green
    img[:30, :, :] = (135, 206, 235)  # sky strip
    return img


def synth_water() -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:30, :, :] = (135, 206, 235)
    img[30:, :, :] = (40, 80, 180)
    return img


def synth_dark_cave() -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, :] = (8, 8, 12)
    # sparse "ore" highlights
    rng = np.random.default_rng(0)
    for _ in range(15):
        x, y = rng.integers(0, 64, size=2)
        img[x, y, :] = (180, 130, 60)
    return img


def synth_trunk_close() -> np.ndarray:
    """Tree trunk close-up: brown vertical band."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, :] = (60, 130, 60)      # background grass
    img[10:60, 18:46, :] = (95, 60, 30)  # brown trunk
    img[8:14, 16:48, :] = (50, 130, 40)  # leaves strip
    return img


def synth_trunk_far() -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:30, :, :] = (135, 206, 235)
    img[30:, :, :] = (60, 130, 60)
    img[28:55, 30:36, :] = (95, 60, 30)
    return img


def to_tensor_224(img64: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert 64x64x3 uint8 → 1x3x224x224 float32 on device."""
    x = torch.from_numpy(img64).float() / 255.0      # (H, W, 3)
    x = x.permute(2, 0, 1).unsqueeze(0)              # (1, 3, H, W)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x.to(device)


def encode_latent(model_inner, frame_uint8_64: np.ndarray, device: torch.device) -> np.ndarray:
    x = to_tensor_224(frame_uint8_64, device)
    with torch.no_grad():
        z = model_inner.encoder(x)                   # (1, 192)
    return z.squeeze(0).cpu().numpy()


def main() -> None:
    print("== OOD probe ==")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout = LatentRollout.from_checkpoint(CKPT, device=device)
    inner = rollout._model._model

    # --- collect 16 real frames + 6 synth frames ---------------------------
    real_frames, real_actions = load_first_npz(SHARD)
    real_frames = real_frames[:16]                       # 16×224×224×3

    # resize real frames to 64x64 for the encoder (matches the agent's
    # 224→64 resize path) and for the OOD probe to be in the same space
    real64 = np.empty((16, 64, 64, 3), dtype=np.uint8)
    for i, fr in enumerate(real_frames):
        # use PIL since it's a resize with good quality
        from PIL import Image
        real64[i] = np.array(Image.fromarray(fr).resize((64, 64), Image.BILINEAR))

    synth = {
        "sky": synth_sky(),
        "ground_close": synth_ground(),
        "water": synth_water(),
        "dark_cave": synth_dark_cave(),
        "trunk_close": synth_trunk_close(),
        "trunk_far": synth_trunk_far(),
    }

    # --- encode all of them ------------------------------------------------
    all_latents: dict[str, np.ndarray] = {}
    for i, fr in enumerate(real64):
        all_latents[f"real_{i:02d}"] = encode_latent(inner, fr, device)
    for name, fr in synth.items():
        all_latents[name] = encode_latent(inner, fr, device)
    names = list(all_latents.keys())
    Z = np.stack([all_latents[n] for n in names])    # (N, 192)
    print(f"latent matrix Z: {Z.shape}")

    # --- norms and pairwise distances -------------------------------------
    norms = np.linalg.norm(Z, axis=-1)
    print("\nlatent L2 norms:")
    for n, v in zip(names, norms):
        print(f"  {n:14s}  ||z|| = {v:7.3f}")
    real_mean_norm = norms[:16].mean()
    real_std_norm = norms[:16].std()
    print(f"\nreal distribution:  mean ||z|| = {real_mean_norm:.3f}  std = {real_std_norm:.3f}")

    # pairwise L2 in latent space
    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    print("\npairwise L2 (selected):")
    cols_of_interest = [names.index(n) for n in ["sky", "ground_close", "water", "trunk_close", "trunk_far"]]
    print(f"  {'':14s}  " + "  ".join([f"{names[c][:10]:>10s}" for c in cols_of_interest]))
    for c in cols_of_interest:
        cname = names[c][:14]
        row = "  ".join([f"{D[c, r]:10.2f}" for r in cols_of_interest])
        print(f"  {cname:14s}  {row}")

    # OOD score for each synth: distance to nearest real
    print("\nOOD: nearest real distance")
    for c in cols_of_interest:
        d = D[c, :16]
        print(f"  {names[c]:14s}  min dist to real: {d.min():.2f}  mean dist to real: {d.mean():.2f}")

    # --- predictor 1-step MSE on real frames ------------------------------
    # Pick a real (frame, action, frame_next) triple.
    print("\n-- 1-step predictor MSE (on a real triple) --")
    f0 = real64[0]; f1 = real64[1]
    a01 = real_actions[0]                                # training-vocab order
    # to_tensor_224 returns (1,3,224,224) — already batched
    f0_t = to_tensor_224(f0, device)                     # (1, 3, 224, 224)
    f1_t = to_tensor_224(f1, device)                     # (1, 3, 224, 224)
    a_t = torch.from_numpy(a01).float().clamp(-1.0, 1.0).to(device).unsqueeze(0)  # (1, 25)
    with torch.no_grad():
        z0 = inner.encoder(f0_t)                         # (1, 192)
        z1 = inner.encoder(f1_t)                         # (1, 192)
        emb0 = inner._projector_fp32(z0)                 # (1, 192)
        emb1 = inner._projector_fp32(z1)
        # Build a 2-frame input so predictor has a length-2 history.
        # Use (emb0, emb0) — predict the next step twice — the change at t=0
        # uses emb0 as current and should predict emb1.
        emb_seq = torch.stack([emb0, emb0], dim=1)       # (1, 2, 192)
        a_emb = inner.action_embedder(torch.stack([a_t, a_t], dim=1))  # (1, 2, 192)
        pred = inner.predictor(emb_seq, a_emb)           # (1, 2, 192)
        change = inner.pred_proj(pred)                   # (1, 2, 192)
        pred_next = emb0 + change[:, 0, :]               # (1, 192) — predicted emb of frame 1
        err_real = (pred_next - emb1).pow(2).sum().item()
    print(f"||pred_emb_1 - real_emb_1||^2 (real triple) = {err_real:.3f}")
    print(f"   ||delta|| = {change[:, 0, :].norm().item():.3f}")
    print(f"   ||emb0|| = {emb0.norm().item():.3f},  ||emb1|| = {emb1.norm().item():.3f}")

    # same but replace f1 with sky
    print("\n-- 1-step predictor: same action, but real f0 -> synth frame target --")
    for name, fr in synth.items():
        fs_t = to_tensor_224(fr, device)   # (1, 3, 224, 224)
        with torch.no_grad():
            zs = inner.encoder(fs_t)
            embs = inner._projector_fp32(zs)
            err = (pred_next - embs).pow(2).sum().item()
        print(f"  {name:14s}  ||pred - synth_emb||^2 = {err:7.2f}  ||synth_emb|| = {embs.norm().item():6.2f}")

    # --- camera action scale probe (Tier 1G) -------------------------------
    # Compare predicted next-latent change magnitude when camera dim is
    # clipped to [-0.3, 0.3] (agent vocab) vs full [-1, 1] (agent vocab).
    # The L0 adapter rescales camera by 180, so the L0 actually sees:
    #   agent_clip 0.3  -> L0 = 0.3 * 180 = 54  (degrees-style)
    #   agent_full  1.0  -> L0 = 180
    print("\n-- action-magnitude probe (camera clip [-0.3,0.3] vs [-1,1] in agent vocab) --")
    a_small = np.zeros(25, dtype=np.float32)
    a_small[0] = 0.3      # camera_pitch
    a_full = np.zeros(25, dtype=np.float32)
    a_full[0] = 1.0
    a_zero = np.zeros(25, dtype=np.float32)
    a_zero[10] = 0.0
    a_big = np.zeros(25, dtype=np.float32)
    a_big[10] = 1.0       # in the L0-vocab (training data) — clamped, so max 1.0

    from wally.planner.rollout import _translate_agent_action_to_l0
    a_small_l0 = _translate_agent_action_to_l0(torch.from_numpy(a_small).to(device))
    a_full_l0 = _translate_agent_action_to_l0(torch.from_numpy(a_full).to(device))
    print(f"  a_small (agent 0.3) -> L0 dim 10 = {a_small_l0[10].item():+.3f}  dim 11 = {a_small_l0[11].item():+.3f}")
    print(f"  a_full  (agent 1.0) -> L0 dim 10 = {a_full_l0[10].item():+.3f}  dim 11 = {a_full_l0[11].item():+.3f}")

    # Now use the L0 predictor via the adapter's predict() (which handles
    # the agent→L0 translation automatically)
    z_curr = torch.from_numpy(all_latents["real_00"]).float().to(device).unsqueeze(0)
    a_small_t = torch.from_numpy(a_small).to(device).unsqueeze(0)
    a_full_t = torch.from_numpy(a_full).to(device).unsqueeze(0)
    a_zero_t = torch.from_numpy(a_zero).to(device).unsqueeze(0)
    a_big_t = torch.from_numpy(a_big).to(device).unsqueeze(0)
    with torch.no_grad():
        d_zero = rollout._model.predict(z_curr, a_zero_t)
        d_small = rollout._model.predict(z_curr, a_small_t)
        d_full = rollout._model.predict(z_curr, a_full_t)
        d_big = rollout._model.predict(z_curr, a_big_t)
    print(f"  ||Δz||  zero camera      : {d_zero.norm().item():.4f}")
    print(f"  ||Δz||  agent camera=0.3 : {d_small.norm().item():.4f}")
    print(f"  ||Δz||  agent camera=1.0 : {d_full.norm().item():.4f}")
    print(f"  ||Δz||  L0 camera=1.0    : {d_big.norm().item():.4f}  (note: this is what training saw)")
    # In training, the L0 saw camera=1.0 (after clamp) = up to 42 deg. But
    # the planner sends agent=1.0 → L0=180. So |Δz_planner| vs |Δz_training|
    # tell us the planner is over-driving the camera by ~4-180x.

    # --- 2D embedding of all frames ---------------------------------------
    print("\n-- 2D embedding (PCA) --")
    X = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(7, 6))
    # real points
    ax.scatter(coords[:16, 0], coords[:16, 1], c="tab:blue", s=40, label="real (16)")
    for i in range(16):
        ax.annotate(f"{i:02d}", (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.6)
    # synth points
    colors = {"sky": "tab:orange", "ground_close": "tab:green", "water": "tab:cyan",
              "dark_cave": "k", "trunk_close": "saddlebrown", "trunk_far": "goldenrod"}
    for n in synth.keys():
        idx = names.index(n)
        ax.scatter(coords[idx, 0], coords[idx, 1], c=colors[n], s=120, marker="X", label=n)
    ax.set_title("L0 latent (PCA-2D): real frames + synth OOD frames")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_ood_latents.png", dpi=120)
    print(f"  wrote {FIG_DIR / '01_ood_latents.png'}")


if __name__ == "__main__":
    main()
