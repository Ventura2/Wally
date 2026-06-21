"""One-shot environment sanity check for the wally project.

Reports the state of the local dev environment and the project's two
run targets (Windows native, WSL2 container) in a single, color-free dump:

- Python version, venv, OS
- torch version + CUDA/ROCm availability
- key wally submodules importable
- entry points in pyproject.toml all resolve to existing modules
- a checkpoint loads via LatentRollout.from_checkpoint
- (when available) the wally-dev Podman container status

Exit code 0 if all required checks pass, 1 otherwise. Optional checks
(e.g. CUDA available, container running) warn but don't fail.

Usage:
    uv run python tools/check_env.py
"""
from __future__ import annotations

import importlib
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _check_python() -> bool:
    _section("Python")
    print(f"  executable : {sys.executable}")
    print(f"  version    : {platform.python_version()}")
    print(f"  prefix     : {sys.prefix}")
    return True


def _check_torch() -> tuple[bool, str]:
    _section("torch")
    try:
        import torch
    except ImportError as exc:
        print(f"  [FAIL] torch not importable: {exc}")
        return False, "missing"
    print(f"  version    : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = "(unavailable)"
        print(f"  cuda       : AVAILABLE ({name})")
        return True, "cuda"
    print("  cuda       : not available (CPU only)")
    if "+cpu" in torch.__version__:
        print("              hint: install TheRock wheels for AMD,")
        print("              see docs/gpu-setup.md")
    return True, "cpu"


def _check_wally_imports() -> bool:
    _section("wally imports")
    ok = True
    for mod in (
        "wally",
        "wally.agent",
        "wally.agent.config",
        "wally.planner",
        "wally.planner.plan",
        "wally.planner.rollout",
    ):
        try:
            importlib.import_module(mod)
            print(f"  [OK]   {mod}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {mod}: {exc}")
            ok = False
    return ok


def _check_entry_points() -> bool:
    _section("entry points (pyproject.toml)")
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    scripts = data.get("project", {}).get("scripts", {})
    ok = True
    for name, ep in scripts.items():
        module = ep.split(":")[0]
        try:
            importlib.import_module(module)
            print(f"  [OK]   {name:30s} -> {ep}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name:30s} -> {ep}  ({exc})")
            ok = False
    return ok


def _check_checkpoint() -> bool:
    _section("checkpoint smoke-load")
    ckpts = sorted(REPO.glob("checkpoints/checkpoint_*.pt"))
    if not ckpts:
        print("  [WARN] no checkpoints/checkpoint_*.pt found")
        return True
    # Prefer the largest (most steps) one for a more thorough test
    target = max(ckpts, key=lambda p: p.stat().st_size)
    print(f"  trying    : {target.name} ({target.stat().st_size // 1024 // 1024} MB)")
    try:
        from wally.planner.rollout import LatentRollout
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] cannot import LatentRollout: {exc}")
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            # Symlink to a short path to avoid Windows MAX_PATH issues
            link = Path(tmp) / target.name
            try:
                link.symlink_to(target)
                rollout = LatentRollout.from_checkpoint(link)
            except OSError:
                rollout = LatentRollout.from_checkpoint(target)
        # Try a forward pass
        import torch
        z0 = torch.zeros(1, 192)
        a = torch.zeros(1, 25)
        with torch.no_grad():
            delta = rollout._model.predict(z0, a)
        print(f"  [OK]   loaded + forward pass: delta shape={tuple(delta.shape)}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        return False


def _check_container() -> bool:
    _section("wally-dev container (optional)")
    if not _command_exists("podman"):
        print("  [SKIP] podman not on PATH")
        return True
    try:
        out = subprocess.run(
            ["podman", "ps", "-a", "--filter", "name=wally-dev",
             "--format", "{{.Names}} {{.Status}} {{.Image}}"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] podman error: {exc}")
        return True
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        print("  [WARN] no wally-dev container found; "
              "see docs/live-viewer.md for setup")
        return True
    for ln in lines:
        print(f"  {ln}")
    if "Up" not in lines[0]:
        print("  [WARN] container is not running; start with: podman start wally-dev")
    return True


def _command_exists(name: str) -> bool:
    return subprocess.run(
        [name, "--version"], capture_output=True, timeout=2
    ).returncode == 0


def main() -> int:
    print(f"Repo: {REPO}")
    print(f"OS  : {platform.platform()}")
    _check_python()
    torch_ok, _ = _check_torch()
    imports_ok = _check_wally_imports()
    ep_ok = _check_entry_points()
    ckpt_ok = _check_checkpoint()
    _check_container()

    _section("summary")
    print(f"  torch           : {'OK' if torch_ok else 'FAIL'}")
    print(f"  wally imports   : {'OK' if imports_ok else 'FAIL'}")
    print(f"  entry points    : {'OK' if ep_ok else 'FAIL'}")
    print(f"  checkpoint load : {'OK' if ckpt_ok else 'FAIL'}")
    if not (imports_ok and ep_ok and ckpt_ok):
        print()
        print("One or more required checks failed. See above.")
        return 1
    print()
    print("All required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
