"""
Rubric Loading Module
====================

Parses ml_rubric.md and loads rules into ChromaDB.

Key Functions:
    - parse_rubric_markdown(path): Parse markdown into rule dicts
    - load_rubric_to_chroma(rules, path): Embed rules and load into ChromaDB
    - load_lecture_chunks_to_chroma(chunks, path): Load lecture chunks into ChromaDB

ChromaDB Collection: rubric_rules
    Schema: {rule_id, topic, subtopic, criteria, socratic_hint, points, ...}
    Embeddings: BAAI/bge-small-en-v1.5 (384-dim)
    Distance metric: Cosine similarity

ChromaDB Collection: lecture_knowledge
    Schema: {text, metadata{source, topic, unit, page, slide_title}}
    Embeddings: BAAI/bge-small-en-v1.5 (384-dim)
    Distance metric: Cosine similarity
"""

import re
import yaml
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def parse_rubric_markdown(rubric_path: str) -> List[Dict]:
    """
    Parse ml_rubric.md into structured rule dictionaries.
    Handles YAML multi-line strings (>, |) and malformed separators robustly.
    
    Args:
        rubric_path: Path to ml_rubric.md file
    
    Returns:
        List of rule dictionaries with fields: rule_id, topic, criteria, socratic_hint, etc.
    """
    with open(rubric_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by '---' (YAML document separator)
    raw_blocks = content.split("---")
    rules = []
    skipped_count = 0
    
    for block_idx, block in enumerate(raw_blocks):
        # Strip leading/trailing whitespace
        block = block.strip()
        
        # Skip empty blocks or blocks that are just dashes
        if not block or block == '-' or re.match(r'^-+$', block.strip()):
            continue
        
        # Remove any leading dashes (malformed separators like '---\n-\nrule_id:')
        if block.startswith('-'):
            block = re.sub(r'^-+\s*', '', block)
        
        if not block.strip():
            continue
        
        try:
            # Parse as YAML (handles multi-line strings with > and |)
            rule = yaml.safe_load(block)
            
            # Skip if parse result is None or empty
            if not rule or not isinstance(rule, dict):
                skipped_count += 1
                continue
            
            # Validate required fields
            required_fields = ["rule_id", "topic", "criteria"]
            missing_fields = [f for f in required_fields if f not in rule]
            
            if missing_fields:
                logger.warning(f"⚠️ Skipping rule (missing {missing_fields}): {rule.get('rule_id', 'UNKNOWN')}")
                skipped_count += 1
                continue
            
            # Success!
            rules.append(rule)
            
        except yaml.YAMLError as e:
            logger.warning(f"❌ YAML parse error in block {block_idx}: {e}")
            skipped_count += 1
        except Exception as e:
            logger.warning(f"❌ Error in block {block_idx}: {e}")
            skipped_count += 1
    
    logger.info(f"✓ Parsed {len(rules)} rubric rules ({skipped_count} skipped)")
    return rules


def load_rubric_to_chroma(rubric_rules: List[Dict], persist_path: str):
    """
    Load rubric rules into ChromaDB as 'rubric_rules' collection.
    Handles duplicate rule IDs by removing duplicates (keeps first occurrence).
    
    Args:
        rubric_rules: List of rule dicts (from parse_rubric_markdown)
        persist_path: Path to ChromaDB persistent storage
    
    Returns:
        ChromaDB collection object
    """
    # Check for duplicates
    rule_ids = [r.get("rule_id") for r in rubric_rules]
    seen = set()
    duplicates = []
    
    for rule_id in rule_ids:
        if rule_id in seen:
            duplicates.append(rule_id)
        seen.add(rule_id)
    
    if duplicates:
        logger.warning(f"⚠️  Found {len(duplicates)} duplicate rule IDs: {duplicates}")
        logger.info(f"   Removing duplicates (keeping first occurrence)...\n")
        
        # Remove duplicates: keep first occurrence only
        seen_ids = set()
        unique_rules = []
        for rule in rubric_rules:
            rule_id = rule.get("rule_id")
            if rule_id not in seen_ids:
                unique_rules.append(rule)
                seen_ids.add(rule_id)
        
        rubric_rules = unique_rules
        logger.info(f"✓ Reduced from {len(rule_ids)} to {len(rubric_rules)} rules\n")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=persist_path)
    
    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection(name="rubric_rules")
        logger.info("✓ Cleared existing rubric_rules collection")
    except:
        pass
    
    # Create collection
    collection = client.get_or_create_collection(
        name="rubric_rules",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize embedder
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    logger.info("✓ BAAI/bge-small-en-v1.5 embedder loaded")
    
    documents = []
    embeddings = []
    metadatas = []
    ids = []
    
    for i, rule in enumerate(rubric_rules):
        # Combine searchable text (topic + subtopic + keywords + criteria)
        embed_text = f"{rule.get('topic', '')} {rule.get('subtopic', '')} " \
                     f"{rule.get('keywords', '')} {rule.get('criteria', '')}"
        
        # Embed
        embedding = embedder.encode(embed_text, convert_to_tensor=False).tolist()
        
        # Prepare for batch insert
        documents.append(rule.get("criteria", ""))
        embeddings.append(embedding)
        metadatas.append({
            "rule_id": rule.get("rule_id", ""),
            "topic": rule.get("topic", ""),
            "subtopic": rule.get("subtopic", ""),
            "points": str(rule.get("points", 0)),
            "common_error": rule.get("common_error", ""),
            "socratic_hint": rule.get("socratic_hint", ""),
            "keywords": rule.get("keywords", "")
        })
        ids.append(rule.get("rule_id", f"rule_{i}"))
        
        if (i + 1) % 20 == 0:
            logger.info(f"Embedded {i + 1}/{len(rubric_rules)} rules...")
    
    # Batch insert into ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info(f"✓ Loaded {len(rubric_rules)} rubric rules into ChromaDB")
    return collection


def load_lecture_chunks_to_chroma(chunks: List[Dict], persist_path: str):
    """
    Load lecture PDF chunks into ChromaDB as 'lecture_knowledge' collection.
    
    Args:
        chunks: List of chunk dicts from extract_slide_chunks
        persist_path: Path to ChromaDB persistent storage
    
    Returns:
        ChromaDB collection object
    """
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(
        name="lecture_knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    documents = []
    embeddings = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        # Embed chunk text
        embedding = embedder.encode(chunk["text"], convert_to_tensor=False).tolist()
        
        # Prepare for insert
        documents.append(chunk["text"])
        embeddings.append(embedding)
        metadatas.append(chunk["metadata"])  # {source, topic, unit, page, slide_title}
        ids.append(f"chunk_{i}")
        
        if (i + 1) % 100 == 0:
            logger.info(f"Embedded {i + 1}/{len(chunks)} lecture chunks...")
    
    # Batch insert
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info(f"✓ Loaded {len(chunks)} lecture chunks into ChromaDB")
    return collection