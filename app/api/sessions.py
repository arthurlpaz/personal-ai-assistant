"""
Session memory endpoints.

GET    /api/sessions              – list all sessions
GET    /api/sessions/{session_id} – get full history for a session
DELETE /api/sessions/{session_id} – delete a session
"""

from fastapi import APIRouter, HTTPException

from app.agent.memory import delete_session, list_sessions, load_session
from app.schemas.chat import HistoryMessage, SessionHistory, SessionInfo

router = APIRouter(prefix="/api", tags=["sessions"])


@router.get("/sessions", response_model=list[SessionInfo])
def get_sessions():
    return list_sessions()


@router.get("/sessions/{session_id}", response_model=SessionHistory)
def get_session_history(session_id: str):
    data = load_session(session_id)
    messages = [HistoryMessage(**m) for m in data.get("messages", [])]
    return SessionHistory(session_id=session_id, messages=messages)


@router.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    ok = delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}
