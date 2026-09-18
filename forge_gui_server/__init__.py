"""ForgeAI desktop GUI backend — FastAPI + WebSocket service layer.

Replaces the PySide6 forge_gui frontend (removed) with a web SPA
(forge_ui/). Reuses the Qt-free domain modules in ``forge_gui.api``
directly and provides async equivalents for the old Qt-coupled services
(process manager, agent loop, GPU polling, engine runtime).
"""
__version__ = "1.0.0"
