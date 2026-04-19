"""FastAPI entry point for the holochat app."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hologram.server import get_or_create_hologram

from .orchestrator import ChatOrchestrator
from .providers import build_provider
from .schemas import ChatRequest, ChatResponse, RouteRequest, RouteResponse


app = FastAPI(
    title="Holochat",
    description="Chatbot application built on top of Hologram retrieval.",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).with_name("static")


def _default_provider():
    if os.getenv("HOLOCHAT_DISABLE_LLM"):
        return None
    return build_provider()


orchestrator = ChatOrchestrator(provider=_default_provider())
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="holochat-assets")


@app.get("/")
async def root():
    return {"status": "running", "service": "holochat", "version": "0.1.0"}


@app.get("/app")
async def app_shell() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat/route", response_model=RouteResponse)
async def route_message(req: RouteRequest) -> RouteResponse:
    try:
        hologram = get_or_create_hologram(req.project)
        route = orchestrator.route(hologram, req.context)
        return RouteResponse(route=route)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/respond", response_model=ChatResponse)
async def respond(req: ChatRequest) -> ChatResponse:
    try:
        hologram = get_or_create_hologram(req.project)
        return orchestrator.respond(
            hologram,
            req.session_id,
            req.context,
            top_k=req.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run("apps.holochat.server:app", host="127.0.0.1", port=8011, reload=False)
