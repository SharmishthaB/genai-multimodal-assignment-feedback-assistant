"""
Retriever Module
================

Unified interface for querying ChromaDB collections with hybrid search.

Key Classes:
    - ChromaRetriever: Query interface with hybrid search (semantic + metadata ranking)
    - RAGSystem: Public API for Dev B agents

Public Methods:
    - retrieve_rubric(topic, error_text, k): Get top-K rubric rules
    - retrieve_context(topic, k): Get top-K lecture chunks

Usage by Dev B:
    from src.rag.retriever import RAGSystem
    
    rag = RAGSystem("/path/to/chromadb")
    rules = rag.retrieve_rubric("Backpropagation", student_work)
    context = rag.retrieve_context("Backpropagation")
"""

import chromadb
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ChromaRetriever:
    """
    Unified interface for querying ChromaDB collections with hybrid search.
    Supports both semantic similarity and metadata-aware ranking.
    """
    
    def __init__(self, persist_path: str):
        """
        Initialize the ChromaRetriever with persistent ChromaDB client.
        
        Args:
            persist_path: Path to ChromaDB persistent storage directory
        
        Raises:
            Exception: If ChromaDB collections cannot be loaded
        """
        self.client = chromadb.PersistentClient(path=persist_path)
        try:
            self.rubric_collection = self.client.get_collection(name="rubric_rules")
            logger.info(f"✓ Rubric collection loaded: {self.rubric_collection.count()} documents")
        except:
            self.rubric_collection = None
            logger.warning("⚠️  Rubric collection not found")
        
        try:
            self.lecture_collection = self.client.get_collection(name="lecture_knowledge")
            logger.info(f"✓ Lecture collection loaded: {self.lecture_collection.count()} documents")
        except:
            self.lecture_collection = None
            logger.warning("⚠️  Lecture collection not found")
    
    def retrieve_rubric(self, topic: str, error_text: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-K rubric rules with HYBRID search:
        1. Semantic search on topic + keywords + criteria
        2. Filter & boost by exact topic match
        3. Re-rank by relevance (3-tier: exact match > related > other)
        
        Args:
            topic: ML topic name (e.g., "Backpropagation")
            error_text: Student's transcribed work or error description
            k: Number of rules to return (default 5)
        
        Returns:
            List[Dict] where each dict contains:
            {
                "rule_id": "BP_001",
                "topic": "Backpropagation",
                "criteria": "Student must apply chain rule...",
                "socratic_hint": "How does error propagate...?",
                "points": "2",
                "distance": 0.15  # Embedding distance (lower = better match)
            }
        
        Example:
            retriever = ChromaRetriever("/kaggle/working/chromadb")
            rules = retriever.retrieve_rubric(
                "Backpropagation", 
                "I applied δ formula to all layers",
                k=3
            )
        """
        if not self.rubric_collection:
            return []
        
        # Construct query: topic is more important for ranking
        query_text = f"Topic: {topic}. Error: {error_text}"
        
        # Retrieve more results initially (top 15) to filter/re-rank by topic
        results = self.rubric_collection.query(
            query_texts=[query_text],
            n_results=min(15, k * 3)  # Get more, then filter
        )
        
        retrieved_rules = []
        if results["ids"] and len(results["ids"]) > 0:
            # Convert results to list of dicts with metadata
            all_results = []
            for idx, doc_id in enumerate(results["ids"][0]):
                rule_dict = {
                    "rule_id": results["metadatas"][0][idx].get("rule_id"),
                    "topic": results["metadatas"][0][idx].get("topic"),
                    "criteria": results["documents"][0][idx],
                    "socratic_hint": results["metadatas"][0][idx].get("socratic_hint"),
                    "points": results["metadatas"][0][idx].get("points"),
                    "distance": results["distances"][0][idx],
                    "keywords": results["metadatas"][0][idx].get("keywords", "")
                }
                all_results.append(rule_dict)
            
            # PRIORITY 1: Exact topic match
            exact_match = [r for r in all_results if r["topic"].lower() == topic.lower()]
            
            # PRIORITY 2: Related topic (e.g., "SVM Hard Margin" for "SVM")
            related_match = [r for r in all_results 
                           if topic.lower().replace(" ", "") in r["topic"].lower().replace(" ", "") 
                           and r not in exact_match]
            
            # PRIORITY 3: Everything else (deprioritized)
            other_match = [r for r in all_results 
                          if r not in exact_match and r not in related_match]
            
            # Combine in order of priority and return top K
            ranked = exact_match + related_match + other_match
            retrieved_rules = ranked[:k]
        
        return retrieved_rules
    
    def retrieve_context(self, topic: str, k: int = 3) -> List[Dict]:
        """
        Retrieve top-K lecture chunks, prioritizing topic matches.
        
        Args:
            topic: ML topic name to search for
            k: Number of context chunks to return (default 3)
        
        Returns:
            List[Dict] where each dict contains:
            {
                "text": "Full slide content...",
                "metadata": {
                    "source": "/path/to/pdf",
                    "topic": "L2_ANN_forward_and_Backpropagation",
                    "page": 5,
                    "unit": "unit2",
                    "slide_title": "Chain Rule in Neural Networks"
                },
                "distance": 0.10
            }
        
        Example:
            retriever = ChromaRetriever("/kaggle/working/chromadb")
            context = retriever.retrieve_context("Backpropagation", k=2)
        """
        if not self.lecture_collection:
            return []
        
        results = self.lecture_collection.query(
            query_texts=[topic],
            n_results=min(10, k * 3)
        )
        
        context_chunks = []
        if results["ids"] and len(results["ids"]) > 0:
            all_chunks = []
            for idx, doc_id in enumerate(results["ids"][0]):
                context = {
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": results["distances"][0][idx]
                }
                all_chunks.append(context)
            
            # Prioritize exact topic match
            exact = [c for c in all_chunks 
                    if c["metadata"].get("topic", "").lower() == topic.lower()]
            other = [c for c in all_chunks if c not in exact]
            
            ranked = exact + other
            context_chunks = ranked[:k]
        
        return context_chunks


class RAGSystem:
    """
    Public API for RAG (Retrieval Augmented Generation) system.
    Used by Dev B's Analyzer and Coach agents for rubric retrieval and context.
    
    This is the main interface that Dev B should use for all retrieval operations.
    """
    
    def __init__(self, chromadb_path: str):
        """
        Initialize the RAG system with ChromaDB path.
        
        Args:
            chromadb_path: Path to ChromaDB persistent storage directory
        """
        from .embedder import EmbedderWrapper
        
        self.retriever = ChromaRetriever(chromadb_path)
        self.embedder = EmbedderWrapper()
        logger.info("✓ RAGSystem initialized")
    
    def retrieve_rubric(self, topic: str, error_text: str, k: int = 5) -> List[Dict]:
        """
        [PUBLIC API] Retrieve relevant rubric rules for an error.
        
        SPEC for Dev B:
        ---------------
        
        Args:
            topic (str): ML topic name (e.g., "Backpropagation")
            error_text (str): Transcribed student work or error description
            k (int): Number of rules to return (default 5)
        
        Returns:
            List[Dict]: Each dict has:
            {
                "rule_id": "BP_001",
                "criteria": "Student must apply chain rule to hidden layers",
                "socratic_hint": "How does error propagate through layer l-1?",
                "points": "2",
                "common_error": "Applying output layer δ formula to all layers",
                "distance": 0.15  # Cosine distance (lower = better match)
            }
        
        Raises:
            ValueError: If topic is empty or error_text is None
        
        Example:
            rag = RAGSystem("/kaggle/working/chromadb")
            rules = rag.retrieve_rubric("Backpropagation", "I used δ=ŷ-y for all layers")
            
            # Use in Analyzer agent:
            analyzer_input = {
                "rules": rules,
                "student_work": transcribed_text,
                "topic": "Backpropagation"
            }
        """
        if not topic or not error_text:
            raise ValueError("topic and error_text cannot be empty")
        
        retrieved = self.retriever.retrieve_rubric(topic, error_text, k=k)
        return retrieved
    
    def retrieve_context(self, topic: str, k: int = 3) -> List[Dict]:
        """
        [PUBLIC API] Retrieve lecture context chunks for a topic.
        
        SPEC for Dev B:
        ---------------
        
        Args:
            topic (str): ML topic name
            k (int): Number of context chunks (default 3)
        
        Returns:
            List[Dict]: Each dict has:
            {
                "text": "Full slide content...",
                "metadata": {
                    "source": "/path/to/pdf",
                    "topic": "Backpropagation",
                    "page": 5,
                    "unit": "unit2",
                    "slide_title": "Chain Rule in Neural Networks"
                },
                "distance": 0.10
            }
        
        Example:
            rag = RAGSystem("/kaggle/working/chromadb")
            context = rag.retrieve_context("Backpropagation")
            coach_prompt = f"Review this context: {context[0]['text']}"
        """
        context = self.retriever.retrieve_context(topic, k=k)
        return context
    
    def get_stats(self) -> Dict:
        """
        [UTILITY] Get system statistics.
        
        Returns:
            Dict with collection counts and model info:
            {
                "rubric_rules_count": 97,
                "lecture_chunks_count": 1176,
                "embedder_model": "BAAI/bge-small-en-v1.5",
                "embedder_dim": 384
            }
        
        Example:
            rag = RAGSystem("/kaggle/working/chromadb")
            stats = rag.get_stats()
            print(f"Rubric rules: {stats['rubric_rules_count']}")
        """
        stats = {
            "rubric_rules_count": self.retriever.rubric_collection.count() if self.retriever.rubric_collection else 0,
            "lecture_chunks_count": self.retriever.lecture_collection.count() if self.retriever.lecture_collection else 0,
            "embedder_model": self.embedder.model_name,
            "embedder_dim": 384
        }
        return stats


# ==============================================================================
# SINGLETON PATTERN: Get RAG system instance (for easy import)
# ==============================================================================

_RAG_INSTANCE = None


def get_rag_system(chromadb_path: str = "/kaggle/working/chromadb") -> RAGSystem:
    """
    Singleton getter for RAGSystem. Dev B can call this from their notebook.
    
    Creates a single RAGSystem instance and returns it on subsequent calls,
    avoiding redundant initialization.
    
    Args:
        chromadb_path: Path to ChromaDB storage (default: Kaggle working directory)
    
    Returns:
        RAGSystem instance (same instance on all calls)
    
    Usage in Dev B's notebook:
        from src.rag.retriever import get_rag_system
        
        rag = get_rag_system()
        rules = rag.retrieve_rubric("Backpropagation", error_text)
        context = rag.retrieve_context("Backpropagation")
    """
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        _RAG_INSTANCE = RAGSystem(chromadb_path)
    return _RAG_INSTANCE
