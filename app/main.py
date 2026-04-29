"""
Application entry point.

Serves the FastAPI backend and the compiled React frontend as static files.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router
from app.api.sessions import router as sessions_router

app = FastAPI(
    title="Personal AI Assistant",
    description="Local-first AI assistant with RAG, memory, and tool calling.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(sessions_router)


@app.get("/health")
def health():
    return {"status": "ok"}


FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend"
    )
