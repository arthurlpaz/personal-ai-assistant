from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agent.rag_tool import semantic_search
from app.agent.tools import ALL_TOOLS
from app.llm.ollama_client import get_llm_with_tools

TOOLS = ALL_TOOLS + [semantic_search]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are a highly capable personal AI assistant running locally.

You have access to the following tools:
- semantic_search: searches the user's personal documents and knowledge base
- calculator: evaluates mathematical expressions
- datetime_info: returns current date and time
- read_file: reads a local text file
- query_csv: summarises a CSV file

Guidelines:
- Always use semantic_search first when the user asks about personal documents, notes, or any topic that might be in their knowledge base.
- Use datetime_info when answering questions about current date/time or scheduling.
- Be concise but thorough. Format responses with markdown when helpful.
- When writing code, use fenced code blocks with the language specified.
- If you don't know something and can't find it with your tools, say so honestly.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph():
    llm_with_tools = get_llm_with_tools(TOOLS)

    def llm_node(state: AgentState) -> dict:
        messages = state["messages"]
        # Inject system prompt if this is the first call
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_fn = TOOL_MAP.get(tool_call["name"])
            if tool_fn is None:
                result = f"Unknown tool: {tool_call['name']}"
            else:
                try:
                    result = tool_fn.invoke(tool_call["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"
            tool_results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        return {"messages": tool_results}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)

    builder.set_entry_point("llm")
    builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "llm")

    return builder.compile()


# Singleton graph (compiled once at import time)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(user_input: str, history: list[dict]) -> tuple[str, list[str]]:
    """
    Run the agent synchronously.
    Returns (final_answer: str, tools_used: list[str]).
    """
    graph = get_graph()

    # Convert history dicts → LangChain message objects
    lc_messages: list[BaseMessage] = []
    for m in history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    lc_messages.append(HumanMessage(content=user_input))

    result = graph.invoke({"messages": lc_messages})

    # Extract final answer
    final_message = result["messages"][-1]
    answer = (
        final_message.content
        if hasattr(final_message, "content")
        else str(final_message)
    )

    # Collect tool names used
    tools_used = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] not in tools_used:
                    tools_used.append(tc["name"])

    return answer, tools_used
