from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's message")
    session_id: str = Field(
        default="default", description="Session identifier for memory"
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    status: str


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    preview: str


class HistoryMessage(BaseModel):
    role: str
    content: str


class SessionHistory(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
