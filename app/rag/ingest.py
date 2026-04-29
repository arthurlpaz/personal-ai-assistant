import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.embeddings import get_embeddings

VECTOR_PATH = "vectorstore"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".rst", ".pdf"}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
)


def _load_file(file_path: str):
    """Load a single file and return a list of Documents."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in {".txt", ".md", ".py", ".rst"}:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


def ingest_file(file_path: str) -> int:
    """
    Ingest a single file into the vectorstore.
    Returns the number of chunks added.
    """
    docs = _load_file(file_path)
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0

    embeddings = get_embeddings()

    if os.path.exists(VECTOR_PATH):
        vs = FAISS.load_local(
            VECTOR_PATH, embeddings, allow_dangerous_deserialization=True
        )
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings)

    vs.save_local(VECTOR_PATH)
    return len(chunks)


def ingest_directory(directory: str = "data") -> dict:
    """
    Ingest all supported files in a directory.
    Returns a summary dict.
    """
    results = {}
    for path in Path(directory).rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            try:
                n = ingest_file(str(path))
                results[str(path)] = {"status": "ok", "chunks": n}
            except Exception as exc:
                results[str(path)] = {"status": "error", "error": str(exc)}
    return results
