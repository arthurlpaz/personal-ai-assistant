"""
Embeddings factory.

Uses nomic-embed-text via Ollama — a dedicated embedding model that
outperforms using a chat model (like Mistral) for embeddings.

Run once to pull the model:
    ollama pull nomic-embed-text
"""

from langchain_ollama import OllamaEmbeddings


def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")
