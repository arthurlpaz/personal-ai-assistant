from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agent.rag_tool import semantic_search
from app.agent.tools import calculator, query_csv, read_file
from app.llm.ollama_client import get_llm


class AgentState(TypedDict):
    input: str
    output: str


llm = get_llm()


def llm_node(state: AgentState):
    response = llm.invoke(state["input"])
    return {"output": response}


def create_graph():
    builder = StateGraph(AgentState)

    builder.add_node("llm", llm_node)

    builder.set_entry_point("llm")
    builder.add_edge("llm", END)

    return builder.compile()
