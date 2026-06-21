"""Doc-freshness tests for the wally project.

These tests grep the project docs for known-bad strings (e.g. stale module
paths that no longer exist) and assert they are absent, and that known-good
canonical strings (e.g. ``python3 -m wally.agent.play``) are present. They
run as part of the standard ``pytest`` suite and catch doc drift before it
trips a human following the quick-start.

If you intentionally change a canonical command path or module name, you
must update both:
- The list of canonical strings asserted present
- The list of stale strings asserted absent
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO / "docs"
LIVE_VIEWER = DOCS_DIR / "live-viewer.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestLiveViewerFreshness:
    """Stale references that were broken on 2026-06-20 and must not return."""

    @pytest.fixture(scope="class")
    def text(self) -> str:
        if not LIVE_VIEWER.is_file():
            pytest.skip(f"missing {LIVE_VIEWER}")
        return _read(LIVE_VIEWER)

    @pytest.mark.parametrize(
        "stale",
        [
            "python3 -m agent.play",
            "pkill -f \"agent.play\"",
            "pkill -TERM -f \"agent.play\"",
            "a/src/agent/loop.py",
            "+++ b/src/agent/loop.py",
        ],
    )
    def test_stale_string_absent(self, text: str, stale: str) -> None:
        assert stale not in text, (
            f"docs/live-viewer.md still contains stale reference {stale!r}; "
            f"remove or update it"
        )

    @pytest.mark.parametrize(
        "canonical",
        [
            "python3 -m wally.agent.play",
            "wally.agent.play",
            "src/wally/agent/loop.py",
            "wally-plan-smoke",
            "tools/start-play-bind.py",
        ],
    )
    def test_canonical_string_present(self, text: str, canonical: str) -> None:
        assert canonical in text, (
            f"docs/live-viewer.md is missing canonical reference {canonical!r}"
        )


class TestEntryPointConsistency:
    """Every script entry point registered in ``pyproject.toml`` should
    import cleanly (i.e. its ``main`` symbol exists and is callable).

    This catches typos in entry-point strings (``"wally.agent.pla:main"``)
    before they ship.
    """

    @pytest.fixture(scope="class")
    def entry_points(self) -> dict[str, str]:
        import tomllib

        with (REPO / "pyproject.toml").open("rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {})
        return {name: ep.split(":")[0] for name, ep in scripts.items()}

    def test_all_entry_point_modules_importable(
        self, entry_points: dict[str, str]
    ) -> None:
        for name, module in entry_points.items():
            try:
                __import__(module)
            except Exception as exc:  # noqa: BLE001
                pytest.fail(
                    f"entry point {name!r} -> {module!r} failed to import: {exc}"
                )

    def test_no_legacy_agent_top_level(self) -> None:
        """``wally-play`` historically pointed at ``agent.play:main``, but
        the package is at ``src/wally/agent/`` - there is no top-level
        ``agent`` package. Make sure no entry point still uses the old
        ``agent.play`` style.
        """
        import tomllib

        with (REPO / "pyproject.toml").open("rb") as f:
            data = tomllib.load(f)
        for name, ep in (data.get("project", {}).get("scripts", {})).items():
            target = ep.split(":")[0]
            assert target != "agent.play", (
                f"entry point {name!r} still points at the non-existent "
                f"top-level package 'agent.play'; use 'wally.agent.play'"
            )
