"""Build the standalone Windows building artifact: ``canary-building.exe``.

Reproducibility notes:
* The interpreter is the repository venv (on Z:), never a system Python.
* PyInstaller's cache, work path, spec path, and temp all live under
  ``outputs/building/`` so nothing is written to C:.
* ``known_templates`` is package data and is collected explicitly.
* ``requests`` is imported lazily by c4nary.remote, so it needs a hidden import;
  the MCP extra is deliberately excluded from the scanner artifact.

Usage:
    .venv\\Scripts\\python.exe tools\\build_exe.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "tools" / "canary_entry.py"
OUT = ROOT / "outputs" / "building"
WORK = OUT / "work"
SPEC = OUT / "spec"
CONFIG = OUT / "pyinstaller-config"
TMP = OUT / "tmp"
EXE = OUT / "canary-building.exe"


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print(
            "PyInstaller is not installed. From the repo root run:\n"
            "  .venv\\Scripts\\python.exe -m pip install -r requirements-build.txt",
            file=sys.stderr,
        )
        return 2

    for path in (OUT, WORK, SPEC, CONFIG, TMP):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYINSTALLER_CONFIG_DIR"] = str(CONFIG)
    env["TEMP"] = str(TMP)
    env["TMP"] = str(TMP)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--console",
        "--clean",
        "--noconfirm",
        "--name", "canary-building",
        "--distpath", str(OUT),
        "--workpath", str(WORK),
        "--specpath", str(SPEC),
        "--paths", str(ROOT),
        "--collect-data", "c4nary",
        "--hidden-import", "requests",
        "--exclude-module", "mcp",
        "--exclude-module", "pytest",
        "--exclude-module", "jsonschema",
        str(ENTRY),
    ]
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if rc != 0:
        return rc
    if not EXE.is_file():
        print(f"build finished but {EXE} is missing", file=sys.stderr)
        return 1
    data = EXE.read_bytes()
    print(f"\nbuilt {EXE} ({len(data):,} bytes)")
    print(f"sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
