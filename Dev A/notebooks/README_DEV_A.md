
# DEV A: RAG + RUBRIC SYSTEM

## Overview
This module implements a Retrieval Augmented Generation (RAG) system for the MAFA project, with two ChromaDB collections:
1. **lecture_knowledge**: ~2000-3000 slide-level chunks from 38 ML lecture PDFs
2. **rubric_rules**: ~130 atomic rubric rules from ml_rubric.md

## Architecture
PDFs (38 files)
↓
[pdf_ingester.py] → extract_slide_chunks() → per-page chunks with metadata
↓
[embedder.py] → BAAI/bge-small-en-v1.5 → 384-dim embeddings
↓
[ChromaDB] → lecture_knowledge collection (persists to /kaggle/working/chromadb/)

ml_rubric.md (synthesized)
↓
[rubric_loader.py] → parse_rubric_markdown() → atomic rules
↓
[embedder.py] → embeddings
↓
[ChromaDB] → rubric_rules collection

[retriever.py] → ChromaRetriever → RAGSystem [PUBLIC API]
↓
[Dev B's Analyzer & Coach agents]


## Key Files

| File | Purpose |
|------|---------|
| `src/ingestion/pdf_ingester.py` | Extract PDF chunks |
| `src/ingestion/rubric_loader.py` | Load rubric rules to ChromaDB |
| `src/rag/retriever.py` | Unified retrieval API |
| `src/rag/embedder.py` | Embedding wrapper |
| `notebooks/dev_a_rag_rubric.ipynb` | Integration notebook |
| `data/rubric/ml_rubric.md` | Synthesized rubric (Section 6 of plan) |

## Public API (for Dev B)

```python
from src.rag.retriever import RAGSystem

# Initialize (singleton)
rag = RAGSystem("/kaggle/working/chromadb")

# Retrieve rubric rules
rules = rag.retrieve_rubric(
    topic="Backpropagation",
    error_text="I applied δ formula to all layers",
    k=5
)
# Returns: [{"rule_id": "BP_001", "criteria": "...", "socratic_hint": "...", ...}]

# Retrieve lecture context
context = rag.retrieve_context(
    topic="Backpropagation",
    k=3
)
# Returns: [{"text": "...", "metadata": {...}, "distance": 0.15}]

# Get system stats
stats = rag.get_stats()
# Returns: {"rubric_rules_count": 130, "lecture_chunks_count": 2847, ...}

Performance Metrics (Day 5)
Hit Rate @ K=3: > 80% on 10 test queries
Hit Rate @ K=5: > 90% on 10 test queries
Collections: Both live and queryable
Embedding Model: BAAI/bge-small-en-v1.5
ChromaDB Location: /kaggle/working/chromadb/

Installation & Setup
# Run in Kaggle Notebook Cell 1
!pip install -q transformers accelerate chromadb sentence-transformers PyMuPDF pdf2image
!apt-get install -q poppler-utils

# Mount Kaggle Dataset containing PDFs
# Expected at: /kaggle/input/ml-lecture-slides/

Troubleshooting
| Issue                     | Solution                                                                 |
|--------------------------|--------------------------------------------------------------------------|
| ChromaDB file not found  | Ensure /kaggle/working/chromadb/ exists; run Day 2 setup                |
| Low Hit Rate @ K         | Check: (1) Rubric loaded correctly, (2) Embedding model consistent, (3) Metadata filters |
| Memory issues            | Both embeddings use CPU; ChromaDB is lightweight                         |
| PDF extraction fails     | Catch exceptions in pdf_ingester.py; log to file for manual inspection   |
Next Steps (Dev B Integration)
1.Import RAGSystem from this module
2.Call retrieve_rubric() in Analyzer agent
3.Call retrieve_context() in Coach agent
4.Design prompts using retrieved rules + context
Timeline
| Day | Task                               | Status      |
|-----|------------------------------------|-------------|
| 1   | PDF extraction + lecture_knowledge | ✓ Complete  |
| 2   | Rubric loading + retriever         | ✓ Complete  |
| 3   | Embedder + Hit Rate @ K            | ✓ Complete  |
| 4   | Public APIs + integration          | ✓ Complete  |
| 5   | Final evaluation + docs            | ✓ Complete  |
Generated: April 2026 | Dev A: RAG System Lead
