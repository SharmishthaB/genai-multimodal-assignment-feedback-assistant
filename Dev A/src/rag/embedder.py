"""
Embedder Module
===============

Wrapper around BAAI/bge-small-en-v1.5 for consistent embeddings.

Key Classes:
    - EmbedderWrapper: Embedding interface

Methods:
    - embed_text(text): Embed single string
    - embed_batch(texts): Embed multiple strings
    - cosine_similarity(vec1, vec2): Compute similarity

Model: BAAI/bge-small-en-v1.5 (384-dim, optimized for retrieval)
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbedderWrapper:
    """
    Wrapper around BAAI/bge-small-en-v1.5 for consistent embedding.
    Provides methods for single text embedding, batch embedding, and similarity computation.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedder with a pre-trained model.
        
        Args:
            model_name: Name of the sentence transformer model to use
                       Default: BAAI/bge-small-en-v1.5 (384-dimensional embeddings)
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"✓ Embedder '{model_name}' initialized")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector (384-dim for bge-small)
        
        Example:
            embedder = EmbedderWrapper()
            emb = embedder.embed_text("What is backpropagation?")
            # Returns List[float] of length 384
        """
        return self.model.encode(text, convert_to_tensor=False).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts efficiently using batch processing.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors (each 384-dim for bge-small)
        
        Example:
            embedder = EmbedderWrapper()
            texts = ["Backpropagation", "Forward pass", "Gradient descent"]
            embs = embedder.embed_batch(texts)
            # Returns List[List[float]], length 3, each element length 384
        """
        return self.model.encode(texts, convert_to_tensor=False).tolist()
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        
        Args:
            vec1: First embedding vector (List[float])
            vec2: Second embedding vector (List[float])
        
        Returns:
            Cosine similarity score (0-1, where 1 = identical, 0 = orthogonal)
        
        Example:
            embedder = EmbedderWrapper()
            v1 = embedder.embed_text("Backpropagation")
            v2 = embedder.embed_text("Chain rule")
            sim = embedder.cosine_similarity(v1, v2)
            # Returns float between 0 and 1
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
