from langchain.tools import tool

from app.rag.retriever import get_retriever

retriever = get_retriever()


@tool
def semantic_search(query: str) -> str:
    """
    Use this tool to search relevant information in personal documents.
    """
    try:
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception:
        return "Error during semantic search"
