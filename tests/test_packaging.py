"""Verifies templates/report.html.jinja actually ships inside the built wheel.

This is a real regression test, not a placeholder: it builds an actual wheel
via `pip wheel` (same command used to manually verify this once) and inspects
the real zip contents, so a future refactor that breaks the
`[tool.setuptools.package-data]` entry in pyproject.toml fails here instead of
failing silently at runtime on whatever machine next runs `ssmlint report`
from an installed (non-editable) package.

The build runs against a *copy* of the repo in a fresh tmp_path, not the
working tree in place -- building in place lets a stale `build/` or
`src/ssmlint.egg-info/` directory (left over from a previous local build)
mask a broken `package-data` glob, since setuptools' incremental build can
reuse a cached file list instead of recomputing it from the current
pyproject.toml. Verified directly: with a leftover build/egg-info present,
this test still passed even after deliberately breaking the glob to
`templates/*.nonexistent`; only a clean-tree build reflects the real config.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COPY_ITEMS = ["pyproject.toml", "src"]


def test_built_wheel_includes_report_template(tmp_path: Path) -> None:
    src_copy = tmp_path / "src_copy"
    src_copy.mkdir()
    for item in _COPY_ITEMS:
        source = _REPO_ROOT / item
        dest = src_copy / item
        if source.is_dir():
            shutil.copytree(source, dest)
        else:
            shutil.copy2(source, dest)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(src_copy), "--no-deps", "--no-cache-dir", "-w", str(out_dir)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"pip wheel failed:\n{result.stdout}\n{result.stderr}"

    wheels = list(out_dir.glob("ssmlint-*.whl"))
    assert len(wheels) == 1, f"expected exactly one built wheel, found {wheels}"

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        assert "ssmlint/templates/report.html.jinja" in names
