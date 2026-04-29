"""
Chat endpoints.

POST /api/chat        – standard synchronous chat
POST /api/chat/stream – streaming via Server-Sent Events (SSE)
"""

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent.graph_agent import run_agent
from app.agent.memory import append_messages, get_history
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = get_history(request.session_id)
    answer, tools_used = run_agent(request.question, history)
    append_messages(request.session_id, request.question, answer)
    return ChatResponse(
        response=answer,
        session_id=request.session_id,
        tools_used=tools_used,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streams the agent response token by token using SSE.
    Each event is a JSON object:
      data: {"token": "...", "done": false}
      data: {"token": "", "done": true, "tools_used": [...]}
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        # Run blocking agent in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        history = get_history(request.session_id)

        answer, tools_used = await loop.run_in_executor(
            None, run_agent, request.question, history
        )

        # Simulate token streaming (real streaming requires LangGraph astream)
        words = answer.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            payload = json.dumps({"token": token, "done": False})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.01)  # slight delay for visual effect

        # Save to memory
        await loop.run_in_executor(
            None, append_messages, request.session_id, request.question, answer
        )

        done_payload = json.dumps({"token": "", "done": True, "tools_used": tools_used})
        yield f"data: {done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
