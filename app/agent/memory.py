"""
Session memory: persists conversation history per session_id as JSON files.
Keeps a sliding window of the last N exchanges to stay within context limits.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

MEMORY_DIR = Path("memory")
MAX_WINDOW = 20  # max messages kept per session (pairs = 10 exchanges)


def _session_path(session_id: str) -> Path:
    MEMORY_DIR.mkdir(exist_ok=True)
    return MEMORY_DIR / f"{session_id}.json"


def load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }


def save_session(session_id: str, data: dict) -> None:
    path = _session_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_history(session_id: str) -> list[dict]:
    """Returns the sliding window of messages for a session."""
    data = load_session(session_id)
    messages = data.get("messages", [])
    return messages[-MAX_WINDOW:]


def append_messages(session_id: str, user_msg: str, assistant_msg: str) -> None:
    data = load_session(session_id)
    data["messages"].append({"role": "user", "content": user_msg})
    data["messages"].append({"role": "assistant", "content": assistant_msg})
    # trim to avoid unbounded growth
    data["messages"] = data["messages"][-(MAX_WINDOW * 2) :]
    data["updated_at"] = datetime.now().isoformat()
    save_session(session_id, data)


def list_sessions() -> list[dict]:
    MEMORY_DIR.mkdir(exist_ok=True)
    sessions = []
    for path in sorted(MEMORY_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        msgs = data.get("messages", [])
        preview = ""
        for m in msgs:
            if m["role"] == "user":
                preview = m["content"][:60]
                break
        sessions.append(
            {
                "session_id": data["session_id"],
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", data.get("created_at", "")),
                "message_count": len(msgs),
                "preview": preview,
            }
        )
    return sessions


def delete_session(session_id: str) -> bool:
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False
