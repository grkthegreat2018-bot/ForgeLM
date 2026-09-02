"""Convenience launcher for the ForgeAI GUI.

Run from the repo root:
    python launch_gui.py
or:
    venv\\Scripts\\python.exe launch_gui.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONUTF8", "1")  # engine prints use → / · (cp1252 breaks)

from forge_gui.app import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run())
