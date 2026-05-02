"""
Long-term semantic memory.

Extracts facts from conversations and stores them in a dedicated FAISS index.
On each new session, retrieves the most relevant facts and injects them into
the system prompt — so the assistant remembers Arthur across sessions.

Fact store: memory/longterm.json  (raw facts + metadata)
Vector store: memory/longterm_vs/ (FAISS index for retrieval)
"""

import json
import re
from datetime import datetime
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.rag.embeddings import get_embeddings

MEMORY_DIR = Path("memory")
FACTS_PATH = MEMORY_DIR / "longterm.json"
VS_PATH = str(MEMORY_DIR / "longterm_vs")

# Extraction prompt — fed to the LLM after each conversation turn
EXTRACTION_PROMPT = """You are a memory extraction system. Given a conversation excerpt, extract factual, reusable information about the user (Arthur Lincoln).

Extract ONLY:
- Personal preferences (tools, languages, workflows he prefers)
- Ongoing projects and their technical details
- Skills and expertise areas
- Goals, deadlines, or commitments mentioned
- Named entities: people, institutions, datasets, models he works with
- Decisions made or conclusions reached

Format as a JSON list of strings. Each fact must be:
- A single, self-contained sentence
- Specific and concrete (not vague like "he likes coding")
- In third person ("Arthur prefers...", "The project uses...")

Example output:
["Arthur works at NUTES, a medical research institution at UFCG in Campina Grande, Brazil.",
 "The ProtesIA project uses TotalSegmentator and nnUNet v2 for CT bone segmentation.",
 "Arthur prefers PyTorch over TensorFlow for deep learning projects."]

If there is nothing worth remembering, return an empty list: []

Conversation:
{conversation}

Return ONLY the JSON list, no other text."""


def _load_facts() -> list[dict]:
    MEMORY_DIR.mkdir(exist_ok=True)
    if not FACTS_PATH.exists():
        return []
    with open(FACTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_facts(facts: list[dict]) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)
    with open(FACTS_PATH, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)


def _rebuild_vectorstore(facts: list[dict]) -> None:
    """Rebuild the FAISS index from the current facts list."""
    if not facts:
        return
    embeddings = get_embeddings()
    docs = [
        Document(
            page_content=f["fact"],
            metadata={
                "source": "longterm_memory",
                "created_at": f.get("created_at", ""),
            },
        )
        for f in facts
    ]
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(VS_PATH)


def _load_vectorstore():
    if not Path(VS_PATH).exists():
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(VS_PATH, embeddings, allow_dangerous_deserialization=True)


def extract_and_store_facts(user_msg: str, assistant_msg: str) -> list[str]:
    """
    Call the LLM to extract facts from a conversation exchange and store them.
    Returns the list of newly extracted facts.
    """
    from app.llm.ollama_client import get_llm

    conversation = f"User: {user_msg}\nAssistant: {assistant_msg}"
    prompt = EXTRACTION_PROMPT.format(conversation=conversation)

    llm = get_llm(temperature=0.0)
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        # Parse JSON — be lenient
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return []
        new_facts: list[str] = json.loads(match.group())
        if not isinstance(new_facts, list):
            return []
        new_facts = [f for f in new_facts if isinstance(f, str) and len(f) > 10]
    except Exception:
        return []

    if not new_facts:
        return []

    existing = _load_facts()
    existing_texts = {f["fact"] for f in existing}

    added = []
    for fact in new_facts:
        if fact not in existing_texts:
            existing.append({"fact": fact, "created_at": datetime.now().isoformat()})
            added.append(fact)

    if added:
        _save_facts(existing)
        _rebuild_vectorstore(existing)

    return added


def retrieve_relevant_facts(query: str, k: int = 6) -> list[str]:
    """
    Retrieve the most relevant long-term facts for a given query.
    Returns a list of fact strings.
    """
    vs = _load_vectorstore()
    if vs is None:
        return []
    try:
        docs = vs.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception:
        return []


def get_all_facts() -> list[dict]:
    return _load_facts()


def delete_fact(index: int) -> bool:
    facts = _load_facts()
    if index < 0 or index >= len(facts):
        return False
    facts.pop(index)
    _save_facts(facts)
    _rebuild_vectorstore(facts)
    return True
