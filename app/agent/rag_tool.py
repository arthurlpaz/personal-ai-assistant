from langchain.tools import tool


@tool
def semantic_search(query: str) -> str:
    """
    Search personal documents for information relevant to the query.
    Use this when the user asks about something that might be covered in
    their uploaded documents, notes, PDFs, or any ingested knowledge base.
    Always try this before saying you don't have information.
    """
    try:
        from app.rag.retriever import get_retriever

        retriever = get_retriever()
        docs = retriever.invoke(
            query
        )  # invoke() replaces deprecated get_relevant_documents()
        if not docs:
            return "No relevant documents found for this query."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[{i}] (source: {source})\n{doc.page_content}")
        return "\n\n---\n\n".join(results)
    except Exception as exc:
        return f"Semantic search unavailable: {exc}"
