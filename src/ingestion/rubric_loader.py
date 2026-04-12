import fitz  # PyMuPDF
from pathlib import Path

def extract_slide_chunks(pdf_path: str) -> list[dict]:
    """Returns one chunk per page with text + metadata."""
    doc = fitz.open(pdf_path)
    chunks = []
    topic = Path(pdf_path).stem  # e.g., "L2_ANN_forward_and_Backpropagation"
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        # Also extract annotation text (Prof's notes on slides)
        annots = [a.info.get("content", "") for a in page.annots() if a.info.get("content")]
        full_text = text + "\n" + "\n".join(annots)
        if len(full_text.strip()) > 50:  # skip blank pages
            chunks.append({
                "text": full_text.strip(),
                "metadata": {
                    "source": str(pdf_path),
                    "topic": topic,
                    "page": page_num + 1,
                    "unit": _detect_unit(pdf_path)
                }
            })
    return chunks

def _detect_unit(path: str) -> str:
    p = str(path).lower()
    for u in ["unit 1", "unit 2", "unit 3", "unit 4"]:
        if u.replace(" ", "") in p.replace(" ", ""):
            return u
    return "unknown"