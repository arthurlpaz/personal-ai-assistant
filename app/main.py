from fastapi import FastAPI

from app.agent.graph_agent import create_graph
from app.schemas.chat import ChatRequest, ChatResponse

app = FastAPI()

graph = create_graph()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = graph.invoke({"input": request.question})
    return ChatResponse(response=result["output"])
