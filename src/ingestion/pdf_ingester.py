"""
PDF Ingestion Module
====================

Extracts text and metadata from lecture PDFs using PyMuPDF (fitz).

Key Functions:
    - extract_slide_chunks(pdf_path): Extract one chunk per slide page
    - _detect_unit(path): Infer unit number from filename
    - _extract_title(text): Extract slide title from content

Chunking Strategy:
    Unit: Per-page chunks
    Metadata: {source, topic, unit, page, slide_title}
    Special handling: Skip blank pages (<50 chars), extract annotations

Output: List[Dict] where each dict has "text" and "metadata" keys
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def extract_slide_chunks(pdf_path: str) -> List[Dict]:
    """
    Extract one chunk per slide page from a PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of dicts with keys: text, metadata

    Raises:
        FileNotFoundError: If PDF doesn't exist
        Exception: If PDF is corrupted
    """
    doc = fitz.open(pdf_path)
    chunks = []
    topic = Path(pdf_path).stem  # e.g., "L2_ANN_forward_and_Backpropagation"

    for page_num, page in enumerate(doc):
        # Extract main text
        text = page.get_text("text")

        # Extract annotations (professor's notes)
        annot_texts = []
        if page.annots():
            for annot in page.annots():
                annot_info = annot.info
                if annot_info.get("content"):
                    annot_texts.append(annot_info["content"])

        # Combine text and annotations
        full_text = text + "\n" + "\n".join(annot_texts)

        # Skip blank pages
        if len(full_text.strip()) > 50:
            # Infer unit from filename
            unit = _detect_unit(str(pdf_path))

            chunks.append({
                "text": full_text.strip(),
                "metadata": {
                    "source": str(pdf_path),
                    "topic": topic,
                    "page": page_num + 1,
                    "unit": unit,
                    "slide_title": _extract_title(text)  # Extract first heading
                }
            })

    doc.close()
    return chunks


def _detect_unit(path: str) -> str:
    """
    Extract unit number from file path.
    
    Args:
        path: File path string
    
    Returns:
        Unit identifier (e.g., "unit1", "unit2") or "unknown"
    """
    p = str(path).lower()
    for u in ["unit 1", "unit 2", "unit 3", "unit 4", "unit1", "unit2", "unit3", "unit4"]:
        if u.replace(" ", "") in p.replace(" ", ""):
            return u.replace(" ", "")
    return "unknown"


def _extract_title(text: str) -> str:
    """
    Extract first line (likely slide title).
    
    Args:
        text: Slide text content
    
    Returns:
        First non-empty line (slide title)
    """
    lines = text.strip().split("\n")
    return lines[0] if lines else ""