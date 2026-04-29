import os
from langchain_community.vectorstores import FAISS
from app.rag.embeddings import get_embeddings

VECTOR_PATH = "vectorstore"


def load_vectorstore():
    embeddings = get_embeddings()

    if not os.path.exists(VECTOR_PATH):
        return None

    return FAISS.load_local(
        VECTOR_PATH, embeddings, allow_dangerous_deserialization=True
    )


def get_retriever(k: int = 4):
    vs = load_vectorstore()
    if vs is None:
        return None
    return vs.as_retriever(search_kwargs={"k": k})
