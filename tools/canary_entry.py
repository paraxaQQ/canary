"""PyInstaller entry point for the standalone building artifact.

PyInstaller needs a script, not a package module with relative imports; this
wrapper is the only build-specific shim.
"""

from __future__ import annotations

import sys

from c4nary.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
