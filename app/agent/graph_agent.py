import threading
from datetime import datetime
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

MAX_TOOL_ROUNDS = 5  # hard cap on tool-call iterations per turn


def build_system_prompt(longterm_facts: list[str] | None = None) -> str:
    now = datetime.now()
    date_str = now.strftime("%A, %d %B %Y — %H:%M")

    facts_block = ""
    if longterm_facts:
        facts_block = (
            "\n\n## What I know about you (from past conversations)\n"
            + "\n".join(f"- {f}" for f in longterm_facts)
        )

    return f"""You are AL Assistant — the personal AI assistant of Arthur Lincoln, an AI/ML Engineer based in Campina Grande, Brazil.

## Current date and time
{date_str}

## About Arthur
- Works at NUTES, a medical research laboratory at UFCG (Federal University of Campina Grande).
- Main project: **ProtesIA** — a CT-based bone segmentation pipeline for prosthetic/implant planning, in partnership with HUAC hospital. Stack: TotalSegmentator, nnUNet v2, PyTorch, pydicom, NiBabel.
- Also works on a **call classification** model (Python/TensorFlow) to predict illegitimate utility calls.
- Fluent in Portuguese and English — respond in whichever language Arthur uses.
- Prefers direct, concise, technically precise responses. No fluff.
{facts_block}

## Tools available
- `semantic_search` — searches Arthur's personal documents and knowledge base. **Always use this first** when asked about any project, document, note, or topic that might be in the knowledge base. If it returns NO_RESULTS, retry with a reformulated query before answering from general knowledge.
- `calculator` — safe math evaluation supporting arithmetic, sqrt, trig, log, etc.
- `datetime_info` — current date, time, weekday, week number.
- `read_file` — reads a local text/code file by path.
- `query_csv` — loads a CSV and returns shape, columns, dtypes, and preview rows.

## Behaviour rules
1. **Think before acting**: decide which tool (if any) is needed. Don't call tools for questions you can answer directly.
2. **RAG first**: for any question about Arthur's work or documents, always search before answering.
3. **Retry on NO_RESULTS**: if `semantic_search` returns NO_RESULTS, try one reformulated query with different keywords before giving up.
4. **Be concise**: use markdown only when it genuinely helps (code blocks, tables, lists). Avoid filler phrases.
5. **Cite sources**: when answering from RAG results, mention the source document name.
6. **Honest uncertainty**: if you don't know and can't find it, say so clearly.
7. **Code quality**: always use fenced code blocks with the language specified. Prefer idiomatic, production-quality code.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_rounds: int


def build_graph():
    llm_with_tools = get_llm_with_tools(TOOLS)

    def llm_node(state: AgentState) -> dict:
        messages = state["messages"]

        # Inject or refresh the system prompt on every call so datetime stays current
        # and long-term facts are included from the first HumanMessage's context
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system:
            # Extract user query to fetch relevant long-term facts
            user_query = ""
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    user_query = m.content
                    break

            try:
                from app.agent.longterm_memory import retrieve_relevant_facts

                facts = retrieve_relevant_facts(user_query, k=6)
            except Exception:
                facts = []

            system = SystemMessage(content=build_system_prompt(facts))
            messages = [system] + list(messages)

        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "tool_rounds": state.get("tool_rounds", 0)}

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

        return {
            "messages": tool_results,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
        }

    def reflect_node(state: AgentState) -> dict:
        """
        Inspect the last ToolMessage. If semantic_search returned NO_RESULTS
        and we haven't hit the cap, inject a coaching message nudging the LLM
        to retry with a reformulated query.
        """
        messages = state["messages"]
        rounds = state.get("tool_rounds", 0)

        # Find the last tool messages
        last_tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if not last_tool_msgs:
            return {}

        last_tool = last_tool_msgs[-1]
        is_empty_rag = last_tool.name == "semantic_search" and (
            "NO_RESULTS" in last_tool.content or len(last_tool.content.strip()) < 30
        )

        if is_empty_rag and rounds < MAX_TOOL_ROUNDS:
            # Nudge the LLM to try a different query
            nudge = HumanMessage(
                content=(
                    "[System: The semantic_search returned no results. "
                    "Please reformulate the query using different keywords or synonyms and try again. "
                    "If after retrying there are still no results, answer from general knowledge.]"
                )
            )
            return {"messages": [nudge]}

        return {}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        rounds = state.get("tool_rounds", 0)

        if rounds >= MAX_TOOL_ROUNDS:
            return END

        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"

        return END

    builder = StateGraph(AgentState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_node("reflect", reflect_node)

    builder.set_entry_point("llm")
    builder.add_conditional_edges(
        "llm",
        should_continue,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "reflect")
    builder.add_edge("reflect", "llm")

    return builder.compile()


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

    After returning, kicks off background fact extraction from this exchange.
    """
    graph = get_graph()

    lc_messages: list[BaseMessage] = []
    for m in history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    lc_messages.append(HumanMessage(content=user_input))

    result = graph.invoke({"messages": lc_messages, "tool_rounds": 0})

    # Extract final answer
    final_message = result["messages"][-1]
    answer = (
        final_message.content
        if hasattr(final_message, "content")
        else str(final_message)
    )

    # Collect unique tool names used
    tools_used = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] not in tools_used:
                    tools_used.append(tc["name"])

    # Background: extract long-term facts from this exchange (non-blocking)
    def _extract():
        try:
            from app.agent.longterm_memory import extract_and_store_facts

            extract_and_store_facts(user_input, answer)
        except Exception:
            pass

    threading.Thread(target=_extract, daemon=True).start()

    return answer, tools_used
