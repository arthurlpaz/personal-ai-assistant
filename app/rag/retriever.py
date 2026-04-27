import os

from langchain_community.vectorstores import FAISS

from app.rag.embeddings import get_embeddings

VECTOR_PATH = "vectorstore"


def load_vectorstore():
    embeddings = get_embeddings()

    if not os.path.exists("vectorstore"):
        print("⚠️ Vectorstore not found. Running ingestion...")
        from app.rag.ingest import ingest

        ingest()

    return FAISS.load_local(
        "vectorstore", embeddings, allow_dangerous_deserialization=True
    )


def get_retriever():
    vs = load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": 3})
