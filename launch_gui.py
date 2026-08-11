"""Convenience launcher for the ForgeAI GUI.

Run from the repo root:
    python launch_gui.py
or:
    venv\\Scripts\\python.exe launch_gui.py
"""
from __future__ import annotations

from forge_gui.app import run

if __name__ == "__main__":
    raise SystemExit(run())
