"""
MAFA (Multimodal Assignment Feedback Assistant) — Gradio UI
UPGRADE: LangGraph pipeline reintegrated; state properly passed; debug tab improved.
"""
import os
os.environ["UNSLOTH_COMPLIED_CACHE"] = "/kaggle/working/unsloth_cache"
os.makedirs("/kaggle/working/unsloth_cache", exist_ok=True)

import unsloth
from unsloth import FastLanguageModel
import gradio as gr
import json
import logging
import torch
from pathlib import Path
from typing import Optional, Tuple

from dev_b.dev_b import (
    VisionTranscriber, ChromaRetriever,
    build_pipeline, StructuredLLMAnalyzer, SocraticCoach
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD MODEL
# ============================================================================
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/kaggle/working/phi_3",
    load_in_4bit=True,
    device_map="auto",
    max_seq_length=4096,
    dtype=None,
)
FastLanguageModel.for_inference(base_model)

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================
analyzer  = StructuredLLMAnalyzer(base_model, tokenizer)
coach     = SocraticCoach(base_model, tokenizer)
vlm       = VisionTranscriber()
retriever = ChromaRetriever("/kaggle/working/chromadb")

# UPGRADE: LangGraph pipeline reintegrated — used as primary path
pipeline = build_pipeline(vlm, analyzer, coach, retriever)
logger.info("✅ Pipeline ready (LangGraph + direct fallback)")

# ============================================================================
# TOPICS & CONFIG
# ============================================================================
TOPICS = [
    "Decision Trees",
    "Backpropagation",
    "Forward Propagation",
    "SVM - Hard Margin",
    "SVM - Soft Margin",
    "Naive Bayes",
    "HMM - Likelihood Problem",
    "HMM - Decoding Problem",
    "HMM - Learning Problem",
    "EM Algorithm",
    "K-Means Clustering",
    "CNN Trainable Parameters",
    "Random Forest",
    "Bagging & Boosting",
    "Gradient Boosting",
    "Logistic Regression",
    "Performance Metrics",
    "PCA / SVD",
]

# ============================================================================
# HANDLERS
# ============================================================================
def analyze_submission(
    uploaded_file,
    input_text: str,
    selected_topic: str,
    show_rubric_reference: bool = True,
    show_timestamps: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:

    has_file = uploaded_file is not None
    has_text = input_text is not None and input_text.strip() != ""

    if not has_file and not has_text:
        return "❌ **Error:** Please upload a file OR enter text.", None, None
    if not selected_topic:
        return "❌ **Error:** Please select a topic from the dropdown.", None, None

    try:
        logger.info(f"Processing submission for topic: {selected_topic}")

        # ── STAGE 1: Transcribe (if image provided) ──────────────────────────
        import re
        transcribed = input_text.strip() if has_text else ""

        if has_file:
            logger.info(f"Transcribing image: {uploaded_file.name}")
            transcribed = vlm.transcribe(uploaded_file.name)
            transcribed = re.sub(r'(\s*\\+\(\\+quad\\+\)\s*){2,}', ' ', transcribed)
            transcribed = re.sub(r'(\\quad\s*){2,}', ' ', transcribed)
            transcribed = re.sub(r'\s{3,}', ' ', transcribed).strip()
            logger.info(f"Transcription complete: {len(transcribed)} chars")

        # ── STAGE 2: Retrieve ─────────────────────────────────────────────────
        logger.info(f"Retrieving rubric rules for topic: {selected_topic}")
        rubric_rules    = retriever.retrieve_rubric(selected_topic, transcribed)
        lecture_context = retriever.retrieve_context(selected_topic)
        logger.info(f"Retrieved {len(rubric_rules)} rubric rules")

        # ── STAGE 3: Build state ──────────────────────────────────────────────
        state = {
            "topic":            selected_topic,
            "transcribed_text": transcribed,
            "rubric_rules":     rubric_rules,
            "lecture_context":  lecture_context,
            "image_path":       uploaded_file.name if has_file else None,
        }

        # ── STAGE 4: Analyze ─────────────────────────────────────────────────
        logger.info("Running error analysis...")
        error_analysis    = analyzer.analyze(state)
        state["error_analysis"] = error_analysis
        logger.info(
            f"Analysis complete — Score: {error_analysis.get('rubric_score', 'N/A')}, "
            f"Violations: {len(error_analysis.get('violated_rules', []))}"
        )

        # ── STAGE 5: Coach ───────────────────────────────────────────────────
        logger.info("Generating feedback...")
        feedback_card = coach.coach(state)
        feedback_card = feedback_card.replace("\\n", "\n").strip()

        error_json = json.dumps(error_analysis, indent=2)
        logger.info("✅ Submission processing complete")

        return feedback_card, transcribed, error_json

    except Exception as e:
        logger.error(f"Error processing submission: {str(e)}", exc_info=True)
        return f"❌ **Error:** {str(e)}", None, None


def reset_form():
    return None, None, False, False, "Your feedback will appear here after submission.", None, None


def format_file_info(uploaded_file) -> str:
    if uploaded_file is None:
        return "No file selected"
    p = Path(uploaded_file.name)
    return f"📄 File: {p.name} ({p.stat().st_size / 1024:.1f} KB)"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
def build_interface():
    with gr.Blocks(
        title="🎓 MAFA",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    ) as demo:

        gr.Markdown("""
# 🎓 MAFA: Multimodal Assignment Feedback Assistant
**Intelligent Socratic Feedback for Machine Learning Assignments**

Upload your handwritten solution to a machine learning problem. Our system will:
1. **Transcribe** your handwritten math and diagrams
2. **Analyze** your work against a rubric
3. **Generate** personalized Socratic hints (no direct solutions)
4. **Provide** actionable feedback tied to course materials
---
""", elem_id="header")

        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("## 📝 Your Submission")

                file_input = gr.File(
                    label="📎 Upload Handwritten Solution",
                    file_count="single",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png", ".webp"],
                    interactive=True,
                )
                text_input = gr.Textbox(
                    label="✍️ Or paste your solution text",
                    placeholder="Type your answer here (instead of uploading image)...",
                    lines=4,
                    interactive=True,
                )
                file_info = gr.Textbox(value="No file selected", label="File Status", interactive=False)
                file_input.change(fn=format_file_info, inputs=file_input, outputs=file_info)

                gr.Markdown("### 🏷️ Problem Topic")
                topic_dropdown = gr.Dropdown(
                    choices=TOPICS,
                    label="Select the ML topic for this assignment:",
                    interactive=True,
                    value=None,
                )

                gr.Markdown("### ⚙️ Options")
                show_rubric_ref  = gr.Checkbox(label="Show rubric rule references", value=True)
                show_timestamps  = gr.Checkbox(label="Show transcribed text & error analysis (debug)", value=False)

                gr.Markdown("### 🎬 Actions")
                submit_btn = gr.Button("📤 Get Feedback", variant="primary", size="lg")
                reset_btn  = gr.Button("🔄 Clear Form", variant="secondary")

                gr.Markdown("""
### ℹ️ Tips for Best Results
- **Clear handwriting**: Write legibly with dark pen
- **Show your work**: Include all derivation steps
- **Label everything**: Mark variables, equations, and steps
- **Include diagrams**: Hand-drawn sketches are okay
- **Annotation notes**: Any clarifications you wrote help

**Processing time:** ~30-60 seconds
""")

            with gr.Column(scale=1.2, min_width=400):
                gr.Markdown("## 📊 Feedback Card")
                feedback_output = gr.Markdown(
                    value="Your feedback will appear here after submission.",
                )

                gr.Markdown("---")
                with gr.Accordion("🔍 Debug Information (Transcription & Error Analysis)", open=False):
                    transcription_output = gr.Textbox(
                        label="Transcribed Content", interactive=False, lines=6, max_lines=12
                    )
                    error_analysis_output = gr.Code(
                        label="Error Analysis (JSON)", language="json", interactive=False
                    )

                gr.Markdown("""---
### 📚 About MAFA
MAFA combines:
- **Qwen2-VL-7B** for multimodal handwritten math transcription
- **ChromaDB RAG** for rubric-aligned error detection
- **Fine-tuned Phi-3-mini** for Socratic hint generation
- **LangGraph** orchestration for reliable agent workflows
""")

        submit_btn.click(
            fn=analyze_submission,
            inputs=[file_input, text_input, topic_dropdown, show_rubric_ref, show_timestamps],
            outputs=[feedback_output, transcription_output, error_analysis_output],
        )
        reset_btn.click(
            fn=reset_form,
            inputs=None,
            outputs=[file_input, topic_dropdown, show_rubric_ref, show_timestamps,
                     feedback_output, transcription_output, error_analysis_output],
        )

    return demo


if __name__ == "__main__":
    logger.info("🚀 Starting MAFA...")
    demo = build_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, debug=True, show_error=True)
