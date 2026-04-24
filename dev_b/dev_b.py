import json
import re
import logging
import os
import torch
import chromadb
from typing import Dict, List, TypedDict, Optional
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LEAKAGE_KEYWORDS = [
    "the answer is", "therefore x =", "the correct formula is", "solving this gives",
    "the solution is", "= [specific numeric answer]", "here's how to solve"
]

PERFECT_SCORE_CHALLENGES = {
    "HMM - Likelihood Problem": "can you explain why the termination step sums over all states rather than taking the maximum?",
    "HMM - Decoding Problem": "can you explain why Viterbi uses max instead of sum, and what goes wrong if you use sum?",
    "HMM - Learning Problem": "can you explain what the E-step and M-step are doing geometrically in Baum-Welch?",
    "Backpropagation": "can you explain what happens to gradients in very deep networks, and how does this affect learning in early layers?",
    "Forward Propagation": "can you trace what happens to the output if one weight in the first layer becomes exactly zero?",
    "Naive Bayes": "can you explain why Laplace smoothing adds |V| to the denominator, not just 1?",
    "SVM - Hard Margin": "can you explain geometrically why only the support vectors determine the decision boundary?",
    "SVM - Soft Margin": "can you explain what increasing C does to the margin width and why?",
    "Decision Trees": "can you explain why information gain tends to favour features with many distinct values?",
    "K-Means Clustering": "can you explain what happens when two centroids converge to the same point?",
    "PCA / SVD": "can you explain the geometric meaning of the first principal component?",
    "Logistic Regression": "can you explain why we use log-loss instead of squared error for classification?",
    "Random Forest": "can you explain how feature bagging reduces correlation between trees?",
    "Gradient Boosting": "can you explain what the residuals in each boosting round represent?",
    "CNN Trainable Parameters": "can you calculate the parameter count if you added one more conv layer with the same settings?",
    "EM Algorithm": "can you explain what the Q function represents and why we maximise it?",
    "Bagging & Boosting": "can you explain why bagging reduces variance but not bias?",
    "Performance Metrics": "can you explain when you would prefer recall over precision, and give a real example?",
}
DEFAULT_CHALLENGE = "can you think of an edge case where this algorithm would behave unexpectedly?"

MODEL_MAX_SEQ_LEN = 4096
ANALYZER_MAX_PROMPT = MODEL_MAX_SEQ_LEN - 500
COACH_MAX_PROMPT = MODEL_MAX_SEQ_LEN - 300


# ──────────────────────────────────────────────────────────────────────────────
# NEW HELPERS — rich rubric formatting (key upgrade)
# ──────────────────────────────────────────────────────────────────────────────
def _format_rubric_for_analyzer(rubric_rules: List[Dict]) -> str:
    """Full detail per rule: criteria + common_error so analyzer has evidence to match."""
    if not rubric_rules:
        return "(No rubric rules retrieved — evaluate based on topic knowledge)"
    lines = []
    for r in rubric_rules:
        rule_id  = r.get("rule_id", "?")
        subtopic = r.get("subtopic", "")
        criteria = r.get("criteria", "").strip()
        points   = r.get("points", "1")
        common_error = r.get("common_error", "").strip()
        line = f"[{rule_id}] {subtopic} ({points} pts)\n  Criteria: {criteria}"
        if common_error:
            line += f"\n  Common Error: {common_error}"
        lines.append(line)
    return "\n\n".join(lines)


def _build_violation_brief(violated_rules: List[Dict], rubric_rules: List[Dict]) -> str:
    rule_map = {
        _clean_rule_id(r.get("rule_id", "")): r
        for r in rubric_rules
    }
    blocks = []
    for i, v in enumerate(violated_rules, 1):
        rid           = _clean_rule_id(v.get("rule_id", ""))
        student_error = v.get("student_error", "").strip()
        severity      = v.get("severity", "minor")
        sev_label     = "critical gap" if severity == "major" else "minor oversight"

        r            = rule_map.get(rid, {})
        subtopic     = r.get("subtopic", "") or rid
        hint_seed    = r.get("socratic_hint", "").strip()
        common_error = r.get("common_error", "").strip()

        block = f"Issue {i} [{rid}] {subtopic} ({sev_label})\n"
        if student_error:
            block += f"  Student error: {student_error}\n"
        if common_error:
            block += f"  Known pattern: {common_error}\n"
        if hint_seed:
            block += f"  Hint seed (rewrite for this student, never copy verbatim): {hint_seed}\n"
        blocks.append(block)

    return "\n".join(blocks) if blocks else "(none)"

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — Schema normaliser: Phi-3 outputs "rule" instead of "rule_id"
# ─────────────────────────────────────────────────────────────────────────────
def _normalise_analysis(raw: Dict, rubric_rules: List[Dict]) -> Dict:
    violations_raw = raw.get("violated_rules", [])
    violations_clean = []
    top_correct_steps = list(raw.get("correct_steps", []))

    for v in violations_raw:
        if not isinstance(v, dict):
            continue

        # ── Deep rescue: model fused keys e.g. "rule_idallacted_rules" ──────
        rescued = []
        bad_keys = []
        for k, val in list(v.items()):
            if (
                k not in {"rule_id", "rule", "student_error", "severity",
                           "correct_steps", "rubric_score"}
                and isinstance(val, list)
            ):
                for item in val:
                    if isinstance(item, dict) and ("rule_id" in item or "rule" in item):
                        rescued.append(item)
                bad_keys.append(k)
        for k in bad_keys:
            del v[k]

        # ── Normalise rule_id: handle "rule" alias + strip full label ────────
        raw_id = str(v.get("rule_id") or v.get("rule", ""))
        v["rule_id"] = _clean_rule_id(raw_id)

        # ── Hoist nested correct_steps / discard nested rubric_score ─────────
        nested = v.pop("correct_steps", [])
        if isinstance(nested, list):
            top_correct_steps.extend(nested)
        v.pop("rubric_score", None)
        v.pop("rule", None)

        # ── Ensure required keys ─────────────────────────────────────────────
        v.setdefault("student_error", "")
        v.setdefault("severity", "minor")

        # Keep only if it carries real content
        has_content = (
            v["rule_id"] != "UNKNOWN"
            and v.get("student_error", "").strip()  # must have actual error text
            and v.get("severity", "").strip()       # must have severity set
        )
        if has_content:
            violations_clean.append(v)

        # ── Add deep-rescued violations ──────────────────────────────────────
        for rv in rescued:
            raw_rv_id = str(rv.get("rule_id") or rv.get("rule", ""))
            rv["rule_id"] = _clean_rule_id(raw_rv_id)
            rv.pop("rule", None)
            rv.setdefault("student_error", "")
            rv.setdefault("severity", "minor")
            if rv["rule_id"] != "UNKNOWN" or rv["student_error"]:
                violations_clean.append(rv)

    raw["violated_rules"] = violations_clean
    raw["correct_steps"] = list(dict.fromkeys(top_correct_steps))
    # ── Clean correct_steps: strip prefixes, split concatenated words ─────────
    cleaned_steps = []
    for s in raw.get("correct_steps", []):
        # Strip "Subtopic:" prefix
        s = re.sub(r"^subtopic:\s*", "", s, flags=re.IGNORECASE).strip()
        # Split CamelCase concatenation e.g. "ConditionalProbabilityCalculation"
        # → "Conditional Probability Calculation"
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s).strip()
        if s:
            cleaned_steps.append(s)
    raw["correct_steps"] = list(dict.fromkeys(cleaned_steps))  # dedup, preserve order
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — Rubric formatter for analyzer (full detail: criteria + common_error)
# ─────────────────────────────────────────────────────────────────────────────
def _format_rubric_for_analyzer(rubric_rules: List[Dict]) -> str:
    if not rubric_rules:
        return "(No rubric rules retrieved)"
    lines = []
    for r in rubric_rules:
        rule_id      = r.get("rule_id", "?")
        subtopic     = r.get("subtopic", "")
        criteria     = r.get("criteria", "").strip()
        points       = r.get("points", "1")
        common_error = r.get("common_error", "").strip()
        line = f"[{rule_id}] {subtopic} ({points} pts)\n  Criteria: {criteria}"
        if common_error:
            line += f"\n  Common Error: {common_error}"
        lines.append(line)
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — Violation brief for coach (robust to normalised rule_id)
# ─────────────────────────────────────────────────────────────────────────────
def _build_violation_brief(violated_rules: List[Dict], rubric_rules: List[Dict]) -> str:
    rule_map = {
        re.sub(r"[\[\]\s]", "", str(r.get("rule_id", ""))): r
        for r in rubric_rules
    }
    blocks = []
    for i, v in enumerate(violated_rules, 1):
        rid_raw       = v.get("rule_id", "")
        rid           = re.sub(r"[\[\]\s]", "", str(rid_raw)) if rid_raw else "UNKNOWN"
        student_error = v.get("student_error", "").strip()
        severity      = v.get("severity", "minor")
        sev_label     = "critical gap" if severity == "major" else "minor oversight"

        r            = rule_map.get(rid, {})
        subtopic     = r.get("subtopic", rid)          # fall back to rule_id as label
        hint_seed    = r.get("socratic_hint", "").strip()
        common_error = r.get("common_error", "").strip()

        block = f"Issue {i} [{rid} — {subtopic}] ({sev_label})\n"
        if student_error:
            block += f"  Student wrote: {student_error}\n"
        if common_error:
            block += f"  Known mistake pattern: {common_error}\n"
        if hint_seed:
            block += f"  Hint seed (adapt, never copy verbatim): {hint_seed}\n"
        blocks.append(block)
    return "\n".join(blocks) if blocks else "(none)"


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — Output cleaner: strips box-drawing / symbol spam from coach output
# ─────────────────────────────────────────────────────────────────────────────
def _clean_generation(text: str) -> str:
    text = re.sub(r"[\u2500-\u257F\u2580-\u259F\u25A0-\u25FF]+", "", text)  # box drawing
    text = re.sub(r"[\u2190-\u21FF\u2200-\u22FF]{2,}", "", text)             # math arrows
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {3,}", " ", text)
    return text.strip()
def _clean_rule_id(rid: str) -> str:
    """
    Normalises any rule_id the model might emit:
      '[HMM_D_004] Viterbi Recursion' → 'HMM_D_004'
      '[DT_002]'                       → 'DT_002'
      'HMM_D_004'                      → 'HMM_D_004'
    """
    cleaned = re.sub(r"[\[\]]", "", str(rid)).strip()
    token = cleaned.split()[0] if cleaned else "UNKNOWN"
    return token if token else "UNKNOWN"


def format_feedback_card(
    topic: str,
    correct_steps: List[str],
    violated_rules: List[Dict],
    rubric_score: int,
    total_pts: int,
    rubric_rules: List[Dict],
    model_hints: str = ""
) -> str:
    """
    Enforces MAFA Implementation Plan feedback card format exactly.
    Never surfaces third-person analyzer text ('The student did not...')
    directly into the card.
    """
    rule_map = {
        _clean_rule_id(r.get("rule_id", "")): r
        for r in rubric_rules
    }

    # ── Header ───────────────────────────────────────────────────────────────
    card = f"## 📋 Feedback Card — {topic}\n\n"

    # ── What you did well ────────────────────────────────────────────────────
    # Strip "Subtopic: " prefix that sometimes leaks from the analyzer
    clean_steps = [re.sub(r"^subtopic:\s*", "", s, flags=re.IGNORECASE).strip()
                   for s in correct_steps]
    # ADD: also split CamelCase
    clean_steps = [re.sub(r"([a-z])([A-Z])", r"\1 \2", s).strip() for s in clean_steps]
    correct_str = ", ".join(clean_steps[:4]) if clean_steps else "some foundational steps"
    card += f"**What you did well:** You correctly handled {correct_str}.\n\n"

    # ── Extract model-generated hints (if clean and question-like) ───────────
    hint_lines = []
    if model_hints:
        cleaned_output = _clean_generation(model_hints)
        splits = re.split(
            r"(?:Issue\s*#?\d+[^:]*:|For\s+Issue\s*#?\d+|Regarding\s+Issue|Hint\s+\d+\s*[\:\(])",
            cleaned_output,
            flags=re.IGNORECASE
        )
        for s in splits:
            candidate = s.strip().split("\n")[0].strip()
            is_clean = (
                len(candidate) > 20
                and len(candidate) < 400
                and "?" in candidate
                and "student" not in candidate.lower()   # block third-person analyzer leakage
                and "the student" not in candidate.lower()
            )
            if is_clean:
                hint_lines.append(candidate)

    # ── Build hints (max 2, major violations first) ──────────────────────────
    sorted_violations = sorted(
        violated_rules,
        key=lambda x: 0 if x.get("severity") == "major" else 1
    )
    top_violations = sorted_violations[:2]

    for idx, v in enumerate(top_violations, 1):
        rid      = _clean_rule_id(v.get("rule_id", ""))
        r        = rule_map.get(rid, {})
        subtopic = r.get("subtopic", "").strip() or rid
        seed     = r.get("socratic_hint", "").strip()
        err      = v.get("student_error", "").strip()

        # Priority: clean model hint → rubric hint seed → rephrased error → generic
        model_hint = hint_lines[idx - 1] if idx - 1 < len(hint_lines) else ""

        if model_hint:
            hint_text = model_hint
        elif seed:
            # Rubric hint seed is the most reliably grounded signal — always prefer it
            hint_text = seed
        elif err:
            # Rephrase third-person analyzer text into second-person Socratic question
            # "The student did not X" → "Did you X? ..."
            rephrased = re.sub(
                r"^the student (did not|didn't|failed to)\s*",
                "Did you ",
                err,
                flags=re.IGNORECASE
            )
            rephrased = re.sub(r"^the student\s+", "You ", rephrased, flags=re.IGNORECASE)
            rephrased = rephrased.rstrip(".")
            hint_text = f"{rephrased}. Can you walk through why this step is required here?"
        else:
            hint_text = (
                f"Revisit the definition of **{subtopic}** from first principles — "
                f"what does it compute, and why is it needed at this stage?"
            )

        card += f"**Hint {idx} (re: {subtopic}):** {hint_text}\n\n"

    # ── Review section ────────────────────────────────────────────────────────
    review_topics = []
    for v in top_violations:
        rid = _clean_rule_id(v.get("rule_id", ""))
        r   = rule_map.get(rid, {})
        t   = (r.get("subtopic") or r.get("topic") or "").strip()
        if t and t not in review_topics:
            review_topics.append(t)
    review_str = ", ".join(review_topics) if review_topics else topic
    card += f"**Review:** Revisit your lecture notes on **{review_str}**.\n\n"

    # ── Rubric Score — clean IDs, no double brackets ──────────────────────────
    ref_ids = ", ".join(
        f"[{_clean_rule_id(str(v.get('rule_id', '?')))}]"
        for v in top_violations
    )
    card += f"**Rubric Score:** {rubric_score} / {total_pts} points  (Ref: Rule {ref_ids})\n"

    return card

# ==================== VISION ====================
class VisionTranscriber:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct", fallback_model_id="OpenGVLab/InternVL2-4B"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.float16,
                quantization_config=bnb_config, device_map="auto"
            )
            self.model.eval()
            logger.info(f"✓ VisionTranscriber loaded: {model_id}")
        except Exception as e:
            logger.warning(f"⚠️ Primary VLM failed: {e}. Falling back to {fallback_model_id}")
            self.processor = AutoProcessor.from_pretrained(fallback_model_id, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                fallback_model_id, torch_dtype=torch.float16,
                quantization_config=bnb_config, device_map="auto", trust_remote_code=True
            )
            self.model.eval()

    def transcribe(self, image_path: str, max_new_tokens: int = 1024) -> str:
        from qwen_vl_utils import process_vision_info
        import re

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": (
                    "This is a student's handwritten solution to a machine learning problem. "
                    "Transcribe ALL content exactly, including: mathematical equations (use LaTeX), "
                    "step labels, numerical values, drawn diagrams described in text, "
                    "and any annotations. Do NOT interpret or correct, only transcribe."
                )}
            ]
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt", padding=True).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        raw = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        cleaned = re.sub(r'(\s*\\+\(\\+quad\\+\)\s*){2,}', ' ', raw)
        cleaned = re.sub(r'(\\quad\s*){2,}', ' ', cleaned)
        cleaned = re.sub(r'\s{3,}', ' ', cleaned)
        return cleaned.strip()




# ==================== RETRIEVER ====================
class ChromaRetriever:
    def __init__(self, path="./chromadb"):
        try:
            self.client = chromadb.PersistentClient(path=path)
            self.rubric = self.client.get_collection("rubric_rules")
            self.lecture = self.client.get_collection("lecture_knowledge")
            logger.info(f"✅ ChromaDB loaded from {path}")
            logger.info(f"   rubric_rules count: {self.rubric.count()}")
            logger.info(f"   lecture_knowledge count: {self.lecture.count()}")
        except Exception as e:
            logger.error(f"❌ ChromaDB load FAILED: {e}")
            self.rubric = self.lecture = None

    def retrieve_rubric(self, topic: str, text: str, k: int = 6) -> List[Dict]:
        if not self.rubric:
            logger.warning("rubric collection is None — returning []")
            return []
        try:
            query_text = f"{topic}: {text[:300]}"
    
            # ── Try with strict topic filter first ─────────────────────────────
            try:
                results = self.rubric.query(
                    query_texts=[query_text],
                    n_results=k,
                    where={"topic": {"$eq": topic}}
                )
                # If filter returned nothing, fall back to unfiltered
                if not results["ids"][0]:
                    logger.warning(f"Topic filter '{topic}' returned 0 results — falling back to unfiltered")
                    raise ValueError("empty filtered result")
            except Exception:
                results = self.rubric.query(
                    query_texts=[query_text],
                    n_results=k
                )
    
            # ── Key fix: ChromaDB stores WITHOUT underscores ────────────────────
            # build_chromadb.py saves: "commonerror", "socratichint"
            # We normalise here so all downstream code uses consistent names
            def _get_meta(m: Dict, *keys) -> str:
                for k in keys:
                    v = m.get(k, "")
                    if v:
                        return str(v)
                return ""
    
            rules = []
            for d, m in zip(results["documents"][0], results["metadatas"][0]):
                rules.append({
                    "rule_id":      _get_meta(m, "rule_id",     "ruleid"),
                    "topic":        _get_meta(m, "topic"),
                    "subtopic":     _get_meta(m, "subtopic"),
                    "criteria":     d,
                    "points":       _get_meta(m, "points") or "1",
                    "socratic_hint": _get_meta(m, "socratic_hint", "socratichint"),
                    "common_error":  _get_meta(m, "common_error",  "commonerror"),
                    "keywords":     _get_meta(m, "keywords"),
                })
    
            logger.info(
                f"retrieve_rubric → {len(rules)} results for topic='{topic}' | "
                f"rule_ids={[r['rule_id'] for r in rules]}"
            )
            return rules
    
        except Exception as e:
            logger.error(f"retrieve_rubric error: {e}")
            return []

    def retrieve_context(self, topic: str, k: int = 3) -> str:
        if not self.lecture:
            logger.warning("lecture collection is None — returning ''")
            return ""
        try:
            results = self.lecture.query(query_texts=[topic], n_results=k)
            chunks = results["documents"][0]
            metas  = results["metadatas"][0]
            parts  = []
            for doc, meta in zip(chunks, metas):
                source = meta.get("source", "")
                page   = meta.get("page", "")
                ref    = f"[{source}, p.{page}]" if source else ""
                parts.append(f"{doc.strip()} {ref}".strip())
            return "\n\n".join(parts)
        except Exception as e:
            logger.error(f"retrieve_context error: {e}")
            return ""


# ==================== ANALYZER ====================
class StructuredLLMAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def analyze(self, state: Dict) -> Dict:
        topic        = state.get("topic", "Unknown Topic")
        topic = topic.strip()   # already there presumably
        # Normalise to title case to match ChromaDB stored values
        # e.g. "backpropagation" → "Backpropagation"
        topic = topic.title() if topic.islower() else topic
        rubric_rules = state.get("rubric_rules", [])
        student_text = state.get("transcribed_text", "").strip()
    
        rubric_text = _format_rubric_for_analyzer(rubric_rules)
    
        prompt = (
            "<|system|>\n"
            f"You are a precise error-detection agent for a Machine Learning course, "
            f"specialised in the topic: **{topic}**.\n\n"
            "Output ONLY a raw JSON object — no markdown fences, no prose — using EXACTLY this schema:\n"
            "{\n"
            '  "topic": "<topic name>",\n'
            '  "violated_rules": [\n'
            '    { "rule_id": "<e.g. NB_004>", "student_error": "<what the student wrote that is wrong>", "severity": "major|minor" }\n'
            "  ],\n"
            '  "correct_steps": ["<subtopic name only, no prefix>"],\n'
            '  "rubric_score": <integer>\n'
            "}\n\n"
            "STRICT RULES:\n"
            '1. The key must be "rule_id" (not "rule"). Do NOT nest correct_steps or rubric_score inside violated_rules.\n'
            "2. ONLY flag a rule as violated if you can quote or closely paraphrase a SPECIFIC mistake "
            "in the student's text. If the student simply did not mention something optional (log trick, "
            "Laplace smoothing, proof derivations) but the core answer is correct, do NOT flag it.\n"
            "3. student_error must describe what the student WROTE that is wrong — not what they omitted "
            "from a textbook treatment.\n"
            "4. correct_steps must be plain subtopic names only — no 'Subtopic:' prefix, no concatenation.\n"
            "5. If the student's final answer is correct, rubric_score must be >= 7 (out of total).\n"
            "<|end|>\n"
            "<|user|>\n"
            f"Topic: {topic}\n\n"
            f"Student's Answer:\n{student_text}\n\n"
            f"Rubric Rules:\n{rubric_text}\n"
            "<|end|>\n"
            "<|assistant|>\n"
        )
    
        inputs    = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
    
        if input_len > ANALYZER_MAX_PROMPT:
            logger.warning(f"⚠️ Analyzer prompt too long ({input_len} tokens). Truncating.")
            inputs["input_ids"]      = inputs["input_ids"][:, -ANALYZER_MAX_PROMPT:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -ANALYZER_MAX_PROMPT:]
            input_len = inputs["input_ids"].shape[1]
    
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
            )
    
        new_tokens = output[0][input_len:]
        res = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
        parsed = None
        try:
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
        except Exception:
            pass
    
        if parsed is None:
            parsed = {"topic": topic, "violated_rules": [], "correct_steps": [], "rubric_score": 0}
    
        # ← KEY: normalise BEFORE resolve_contradictions
        parsed = _normalise_analysis(parsed, rubric_rules)
        parsed = self._resolve_contradictions(parsed, rubric_rules)
        return parsed

    def _resolve_contradictions(self, analysis: Dict, rubric_rules: List[Dict]) -> Dict:
        if not rubric_rules:
            logger.warning("_resolve_contradictions called with empty rubric_rules")
            analysis["rubric_score"] = 0
            analysis["correct_steps"] = []
            return analysis
    
        topic = analysis.get("topic", "")
    
        # ── Drop cross-topic violations ───────────────────────────────────────────
        valid_prefixes = set()
        for r in rubric_rules:
            rid = _clean_rule_id(r.get("rule_id", ""))
            prefix = re.match(r"^[A-Z]+", rid)
            if prefix:
                valid_prefixes.add(prefix.group())
    
        if valid_prefixes:
            filtered = []
            for v in analysis.get("violated_rules", []):
                rid    = _clean_rule_id(v.get("rule_id", ""))
                prefix = re.match(r"^[A-Z]+", rid)
                if prefix and prefix.group() not in valid_prefixes:
                    logger.warning(f"Dropping cross-topic violation: {rid} in topic '{topic}'")
                    continue
                filtered.append(v)
            analysis["violated_rules"] = filtered
    
        # ── Drop contradictions — ONLY if overlap is very high (raised from 0.4 → 0.6) ──
        # Lower threshold was too aggressive, dropping real violations
        step_text = " ".join(analysis.get("correct_steps", [])).lower()
        cleaned_violations = []
        for v in analysis.get("violated_rules", []):
            if v.get("severity") == "none":
                continue
            err_text  = v.get("student_error", "").lower()
            err_words = {w for w in err_text.split() if len(w) > 5}  # raised min word len 4→5
            if err_words and step_text:
                overlap = sum(1 for w in err_words if w in step_text)
                if overlap / len(err_words) > 0.6:   # raised threshold 0.4 → 0.6
                    logger.info(f"Dropping contradictory violation {v.get('rule_id')}")
                    continue
            cleaned_violations.append(v)
        analysis["violated_rules"] = cleaned_violations
    
        # ── Cap violations at 3 to prevent hallucination pile-on ─────────────────
        # Sort: major first, then keep at most 3
        cleaned_violations = sorted(
            cleaned_violations,
            key=lambda x: 0 if x.get("severity") == "major" else 1
        )[:3]
        analysis["violated_rules"] = cleaned_violations
    
        # ── Recompute rubric score ────────────────────────────────────────────────
        rule_map = {
            _clean_rule_id(r.get("rule_id", "")): r
            for r in rubric_rules
        }
        total_possible = sum(int(r.get("points", 1)) for r in rubric_rules)
        points_lost = sum(
            int(rule_map.get(_clean_rule_id(v.get("rule_id", "")), {}).get("points", 1))
            for v in cleaned_violations
        )
        analysis["rubric_score"] = max(0, total_possible - points_lost)
    
        # ── Fill correct_steps if empty ───────────────────────────────────────────
        violated_ids = {_clean_rule_id(v.get("rule_id", "")) for v in cleaned_violations}
        if not analysis.get("correct_steps"):
            analysis["correct_steps"] = [
                r.get("subtopic") or r.get("criteria", "")[:60]
                for r in rubric_rules
                if _clean_rule_id(r.get("rule_id", "")) not in violated_ids
            ]
    
        logger.info(
            f"Score: {analysis['rubric_score']}/{total_possible} "
            f"(violations after cap: {len(cleaned_violations)}, total rules: {len(rubric_rules)})"
        )
        return analysis


# ==================== COACH ====================
class SocraticCoach:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def coach(self, state: Dict) -> str:
        topic           = state.get("topic", "Unknown Topic")
        transcribed     = state.get("transcribed_text", "").strip()
        error_analysis  = state.get("error_analysis", {})
        rubric_rules    = state.get("rubric_rules", [])
        lecture_context = state.get("lecture_context", "").strip()
    
        violated_rules = error_analysis.get("violated_rules", [])
        correct_steps  = error_analysis.get("correct_steps", [])
        rubric_score   = error_analysis.get("rubric_score", 0)
        total_pts      = sum(int(r.get("points", 1)) for r in rubric_rules) if rubric_rules else 10
    
        # ── Perfect path ───────────────────────────────────────────────────────
        if not violated_rules:
            challenge = PERFECT_SCORE_CHALLENGES.get(topic, DEFAULT_CHALLENGE)
            return (
                f"## ✅ Excellent Work on {topic}!\n\n"
                f"Your solution correctly addresses all the key rubric criteria. "
                f"Score: **{rubric_score}/{total_pts}**\n\n"
                f"**To deepen your understanding:** {challenge}\n\n"
                f"*Try explaining your reasoning to a peer — teaching consolidates understanding.*"
            )
    
        # ── Build structured brief (now works because rule_id is normalised) ───
        violation_brief = _build_violation_brief(violated_rules, rubric_rules)
        lecture_snippet = (
            f"\nRelevant course context:\n{lecture_context[:400]}\n"
            if lecture_context else ""
        )
        correct_summary = ", ".join(correct_steps[:4]) if correct_steps else "some foundational steps"
    
        prompt = (
            "<|system|>\n"
            f"You are MAFA, a Socratic ML tutor specialised in **{topic}**.\n\n"
            "For each numbered issue below, write exactly ONE Socratic question (2-3 sentences max).\n"
            "Rules:\n"
            "- Reference the student's actual mistake in your question (use their words).\n"
            "- For 'critical gap': ask a multi-step reasoning question (what → why → implication).\n"
            "- For 'minor oversight': ask one targeted question.\n"
            "- Use the hint seed as inspiration — rewrite it specific to this student's error.\n"
            "- NEVER give the correct answer, formula, or value.\n"
            "- Stop after addressing all listed issues. Do not add filler text.\n"
            "<|end|>\n"
            "<|user|>\n"
            f"Topic: {topic}\n"
            f"Student's solution (excerpt):\n{transcribed[:500]}\n\n"
            f"What the student got right: {correct_summary}\n"
            f"Score: {rubric_score}/{total_pts}\n\n"
            f"Issues to address:\n{violation_brief}"
            f"{lecture_snippet}"
            "<|end|>\n"
            "<|assistant|>\n"
            f"## 📝 Feedback on Your {topic} Solution\n\n"
            f"Good effort — you correctly handled: **{correct_summary}**.\n\n"
            f"**Score: {rubric_score}/{total_pts}** — here are some questions to guide your review:\n\n"
        )
    
        inputs    = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
    
        if input_len > COACH_MAX_PROMPT:
            logger.warning(f"⚠️ Coach prompt too long ({input_len} tokens). Truncating.")
            inputs["input_ids"]      = inputs["input_ids"][:, -COACH_MAX_PROMPT:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -COACH_MAX_PROMPT:]
            input_len = inputs["input_ids"].shape[1]
    
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=400,           # hard cap — prevents runaway generation
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,       # stronger than before (was 1.15)
                eos_token_id=self.tokenizer.eos_token_id,
            )
    
        new_tokens = output[0][input_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
        # Strip symbol spam (box-drawing, math arrows from degenerate loops)
        raw = _clean_generation(raw)

        # Enforce MAFA spec format — don't trust model structure
        feedback_card = format_feedback_card(
            topic          = topic,
            correct_steps  = correct_steps,
            violated_rules = violated_rules,
            rubric_score   = rubric_score,
            total_pts      = total_pts,
            rubric_rules   = rubric_rules,
            model_hints    = raw          # pass raw model output for hint extraction
        )
        return feedback_card


# ==================== PIPELINE ====================
class PipelineState(TypedDict, total=False):
    image_path: Optional[str]
    topic: str
    transcribed_text: str
    rubric_rules: List[Dict]
    lecture_context: str
    error_analysis: Dict
    feedback_card: str


def build_pipeline(vlm, analyzer, coach, retriever=None):
    graph = StateGraph(PipelineState)
    _retriever = retriever or ChromaRetriever()

    def n_transcribe(s):
        return {"transcribed_text": vlm.transcribe(s["image_path"])}

    def n_passthrough(s):
        return {}

    def n_retrieve(s):
        logger.info(f"n_retrieve called — topic='{s.get('topic')}' transcribed_len={len(s.get('transcribed_text', ''))}")
        rules = _retriever.retrieve_rubric(s["topic"], s.get("transcribed_text", ""))
        ctx   = _retriever.retrieve_context(s["topic"])
        logger.info(f"n_retrieve returning {len(rules)} rules, context_len={len(ctx)}")
        return {"rubric_rules": rules, "lecture_context": ctx}

    def n_analyze(s):
        return {"error_analysis": analyzer.analyze(s)}

    def n_coach(s):
        return {"feedback_card": coach.coach(s)}

    def route_input(s):
        return "transcribe" if s.get("image_path") else "passthrough"

    graph.add_node("transcribe", n_transcribe)
    graph.add_node("passthrough", n_passthrough)
    graph.add_node("retrieve",   n_retrieve)
    graph.add_node("analyze",    n_analyze)
    graph.add_node("coach",      n_coach)

    graph.set_conditional_entry_point(
        route_input,
        {"transcribe": "transcribe", "passthrough": "passthrough"}
    )
    graph.add_edge("transcribe",  "retrieve")
    graph.add_edge("passthrough", "retrieve")
    graph.add_edge("retrieve",    "analyze")
    graph.add_edge("analyze",     "coach")
    graph.add_edge("coach",       END)

    return graph.compile()


# ==================== METRICS & VALIDATION ====================
def validate_analysis_structure(analysis: Dict) -> bool:
    required = ["topic", "violated_rules", "correct_steps", "rubric_score"]
    return (
        all(f in analysis for f in required)
        and isinstance(analysis["violated_rules"], list)
    )


def cer(predicted: str, ground_truth: str) -> float:
    if not ground_truth:
        return 1.0 if predicted else 0.0
    s1, s2 = predicted.replace(" ", ""), ground_truth.replace(" ", "")

    def lev(a, b):
        if len(a) < len(b):
            return lev(b, a)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, c1 in enumerate(a):
            curr = [i + 1]
            for j, c2 in enumerate(b):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    return lev(s1, s2) / len(s2)


def solution_leakage_rate(cards: List[str]) -> float:
    if not cards:
        return 0.0
    leaks = sum(1 for c in cards if any(kw in c.lower() for kw in LEAKAGE_KEYWORDS))
    return leaks / len(cards)


def get_pipeline_metrics() -> Dict:
    return {
        "solution_leakage_rate": solution_leakage_rate,
        "cer": cer,
        "validate_analysis_structure": validate_analysis_structure,
    }
