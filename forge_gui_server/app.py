"""ForgeAI GUI backend — FastAPI app: REST API + WebSocket event stream
+ static SPA hosting for forge_ui/dist.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .deps import services
from .hub import dumps
from .routes import chat, core, system, train

logger = logging.getLogger(__name__)

_UI_DIST = Path(__file__).resolve().parents[1] / "forge_ui" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await services.start()
    yield
    await services.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="ForgeAI GUI", version="1.0.0",
                  lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
        allow_headers=["*"])

    app.include_router(core.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(train.router, prefix="/api")
    app.include_router(system.router, prefix="/api")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        q = services.hub.subscribe(replay=True)
        try:
            while True:
                evt = await q.get()
                await websocket.send_text(dumps(evt))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            services.hub.unsubscribe(q)

    # ── SPA static hosting ────────────────────────────────────────────
    if _UI_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=_UI_DIST / "assets"),
                  name="assets")

        @app.get("/{full_path:path}")
        async def spa(full_path: str):
            # SPA fallback — every non-API route serves index.html
            if full_path.startswith(("api/", "ws")):
                return {"error": "not found"}
            index = _UI_DIST / "index.html"
            return FileResponse(index)

    return app


app = create_app()
