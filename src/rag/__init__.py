"""
RAG Module
==========

Retrieval Augmented Generation system for the MAFA project.

Modules:
    - embedder: BAAI/bge-small-en-v1.5 embedding wrapper
    - retriever: ChromaDB retriever with hybrid search and public RAGSystem API

Key Classes:
    - EmbedderWrapper: Embeddings interface
    - ChromaRetriever: Internal retrieval implementation
    - RAGSystem: Public API for Dev B
"""

from .embedder import EmbedderWrapper
from .retriever import ChromaRetriever, RAGSystem, get_rag_system

__all__ = [
    "EmbedderWrapper",
    "ChromaRetriever",
    "RAGSystem",
    "get_rag_system"
]
