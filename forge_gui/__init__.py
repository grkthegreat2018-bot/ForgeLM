"""ForgeAI GUI domain layer — Qt-free service modules.

The PySide6 desktop shell was replaced by ``forge_gui_server`` (FastAPI +
pywebview) and the React frontend in ``forge_ui/``. What remains here is
``forge_gui.api``: plain-Python domain modules shared with the server
(status readers, chat store, lorebook, tool harness, managers).
"""
from __future__ import annotations

__version__ = "0.1.0"
