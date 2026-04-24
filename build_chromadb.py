"""
Standalone ChromaDB builder — run in its own process to avoid torch/unsloth conflicts.
"""
import os, re, yaml, json, fitz, chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "/kaggle/working/chromadb"
RUBRIC_PATH = "/kaggle/input/datasets/sharmishthab/mafa-rubrics/ml_rubric.md"
PDF_DIR     = "/kaggle/input/datasets/sharmishthab/ml-slides"

os.makedirs(CHROMA_PATH, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_PATH)

# ── Skip if already built ──────────────────────────────────
try:
    rc = client.get_collection("rubric_rules")
    lc = client.get_collection("lecture_knowledge")
    if rc.count() > 0 and lc.count() > 0:
        print(f"✅ Already built — rubric_rules: {rc.count()}, lecture_knowledge: {lc.count()}")
        exit(0)
except:
    pass

print("⏳ Building ChromaDB...")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

# ── 1. Rubric Collection ───────────────────────────────────
def parse_rubric_markdown(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    rules = []
    for block in content.split("---"):
        block = block.strip()
        if not block or re.match(r'^-+$', block): continue
        if block.startswith('-'):
            block = re.sub(r'^-+\s*', '', block)
        if not block: continue
        try:
            rule = yaml.safe_load(block)
            if rule and isinstance(rule, dict) and all(f in rule for f in ["rule_id","topic","criteria"]):
                rules.append(rule)
        except: continue
    return rules

try: client.delete_collection("rubric_rules")
except: pass
rubric_col = client.create_collection("rubric_rules", metadata={"hnsw:space":"cosine"})

rules = parse_rubric_markdown(RUBRIC_PATH)
seen, unique = set(), []
for r in rules:
    if r["rule_id"] not in seen:
        unique.append(r); seen.add(r["rule_id"])
rules = unique

docs, embs, metas, ids = [], [], [], []
for i, r in enumerate(rules):
    txt = f"{r.get('topic','')} {r.get('subtopic','')} {r.get('keywords','')} {r.get('criteria','')}"
    docs.append(r.get("criteria",""))
    embs.append(embedder.encode(txt, convert_to_tensor=False).tolist())
    metas.append({k: str(r.get(k,"")) for k in ["rule_id","topic","subtopic","points","common_error","socratic_hint","keywords"]})
    ids.append(str(r.get("rule_id", f"rule_{i}")))

rubric_col.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
print(f"✅ rubric_rules: {rubric_col.count()} documents")

# ── 2. Lecture Collection ──────────────────────────────────
try: client.delete_collection("lecture_knowledge")
except: pass
lecture_col = client.create_collection("lecture_knowledge", metadata={"hnsw:space":"cosine"})

all_chunks, l_docs, l_metas, l_ids = [], [], [], []
for pdf_path in sorted(Path(PDF_DIR).glob("*.pdf")):
    try:
        doc = fitz.open(str(pdf_path))
        topic = pdf_path.stem
        for pn, page in enumerate(doc):
            text = page.get_text("text")
            annots = [a.info.get("content","") for a in (page.annots() or []) if a.info.get("content")]
            full = (text + "\n" + "\n".join(annots)).strip()
            if len(full) > 50:
                all_chunks.append({"text": full[:2000], "meta": {
                    "topic": topic, "page": str(pn+1),
                    "source": pdf_path.name,
                    "slide_title": full.split("\n")[0][:100]
                }})
        doc.close()
    except Exception as e:
        print(f"  ⚠️ Skipped {pdf_path.name}: {e}")

BATCH = 64
for i in range(0, len(all_chunks), BATCH):
    batch = all_chunks[i:i+BATCH]
    texts = [c["text"] for c in batch]
    batch_embs = embedder.encode(texts, batch_size=BATCH, convert_to_tensor=False).tolist()
    lecture_col.add(
        documents=texts,
        embeddings=batch_embs,
        metadatas=[c["meta"] for c in batch],
        ids=[f"chunk_{i+j}" for j in range(len(batch))]
    )
    if (i+BATCH) % 200 < BATCH:
        print(f"  Embedded {min(i+BATCH, len(all_chunks))}/{len(all_chunks)} chunks...")

print(f"✅ lecture_knowledge: {lecture_col.count()} documents")
print("🎉 ChromaDB build complete!")
