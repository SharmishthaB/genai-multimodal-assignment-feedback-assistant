"""
Ingestion Module
================

Tools for extracting and loading lecture content and rubric rules.

Modules:
    - pdf_ingester: PDF text extraction with metadata
    - rubric_loader: YAML-based rubric parsing and ChromaDB loading
"""

from .pdf_ingester import extract_slide_chunks, _detect_unit, _extract_title
from .rubric_loader import (
    parse_rubric_markdown,
    load_rubric_to_chroma,
    load_lecture_chunks_to_chroma
)

__all__ = [
    "extract_slide_chunks",
    "_detect_unit",
    "_extract_title",
    "parse_rubric_markdown",
    "load_rubric_to_chroma",
    "load_lecture_chunks_to_chroma"
]
