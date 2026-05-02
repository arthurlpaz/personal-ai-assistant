import json
import re

from langchain.tools import tool
from langchain_core.documents import Document


def _expand_query(query: str) -> list[str]:
    """Generate alternative phrasings for the query using the LLM."""
    from app.llm.ollama_client import get_llm

    prompt = f"""Generate 3 alternative search queries for the following question.
Each variant should use different vocabulary but seek the same information.
Return ONLY a JSON list of 3 strings, nothing else.

Original query: {query}

"""

    llm = get_llm(temperature=0.3)
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            variants = json.loads(match.group())
            if isinstance(variants, list):
                return [str(v) for v in variants[:3]]
    except Exception:
        pass
    return []


def _deduplicate(docs: list[Document]) -> list[Document]:
    """Remove near-duplicate chunks (same first 80 chars)."""
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content[:80].strip()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _format_results(docs: list[Document]) -> str:
    if not docs:
        return "NO_RESULTS"
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        source = source.split("/")[-1] if "/" in source else source
        parts.append(f"[{i}] source: {source}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


@tool
def semantic_search(query: str) -> str:
    """
    Search Arthur's personal knowledge base (uploaded documents, notes, PDFs).

    Use this tool when the user asks about:
    - Their own documents, projects, notes, or research
    - Technical details of any project (ProtesIA, call classification, etc.)
    - Any topic that might be covered in uploaded files

    The tool automatically expands the query to improve recall.
    If results say NO_RESULTS, the knowledge base may be empty or the topic
    is not covered — inform the user and answer from general knowledge.
    """
    from app.rag.retriever import get_retriever

    retriever = get_retriever(k=4)
    if retriever is None:
        return "Knowledge base is empty. Ask the user to upload documents first."

    try:
        primary_docs = retriever.invoke(query)
    except Exception as exc:
        return f"Search error: {exc}"

    extra_docs: list[Document] = []
    try:
        variants = _expand_query(query)
        for variant in variants:
            try:
                docs = retriever.invoke(variant)
                extra_docs.extend(docs)
            except Exception:
                pass
    except Exception:
        pass

    all_docs = _deduplicate(primary_docs + extra_docs)[:5]
    return _format_results(all_docs)
