"""
MAFA UI Configuration
Visual, behavioral, and functional settings for the Gradio interface.

File: config/ui_config.json (or hardcoded in app.py)
Used by: src/ui/app.py
Updated: 2025-04-14
"""

import json
from typing import Dict, List, Any
from pathlib import Path

# ============================================================================
# UI CONFIGURATION
# ============================================================================

UI_CONFIG = {
    # ========================================================================
    # BRANDING & METADATA
    # ========================================================================
    "app_name": "MAFA",
    "app_full_name": "Multimodal Assignment Feedback Assistant",
    "app_description": "Intelligent Socratic feedback system for machine learning assignments",
    "version": "1.0.0",
    "created_date": "2025-04-14",
    
    # ========================================================================
    # VISUAL THEME & COLORS
    # ========================================================================
    "theme": {
        "name": "Soft",
        "primary_color": "#0084d6",      # Education blue
        "primary_hue": "blue",
        "secondary_hue": "slate",
        "accent_color": "#00a86b",       # Success green
        "warning_color": "#ff9800",      # Warning orange
        "error_color": "#f44336",        # Error red
        "background": "light",
        "font_family": "system-ui, -apple-system, sans-serif",
        "border_radius": "8px",
        "shadow_depth": "medium",
    },
    
    # ========================================================================
    # HEADER & BRANDING
    # ========================================================================
    "header": {
        "title": "🎓 MAFA: Multimodal Assignment Feedback Assistant",
        "subtitle": "Intelligent Socratic Feedback for Machine Learning Assignments",
        "description": (
            "Upload your handwritten solution to a machine learning problem. Our system will:\n"
            "1. **Transcribe** your handwritten math and diagrams\n"
            "2. **Analyze** your work against a comprehensive rubric\n"
            "3. **Generate** personalized Socratic hints (no direct solutions)\n"
            "4. **Provide** actionable feedback tied to course materials"
        ),
        "show_divider": True,
    },
    
    # ========================================================================
    # FILE UPLOAD WIDGET
    # ========================================================================
    "file_upload": {
        "label": "📎 Upload Handwritten Solution",
        "file_count": "single",
        "file_types": [".pdf", ".jpg", ".jpeg", ".png", ".webp"],
        "max_size_mb": 50,
        "accepted_formats": [
            "image/jpeg",
            "image/png",
            "image/webp",
            "application/pdf"
        ],
        "help_text": (
            "Supported formats: PDF, JPG, PNG, WEBP\n"
            "Maximum file size: 50 MB\n"
            "Best quality: High contrast, clear handwriting"
        ),
    },
    
    # ========================================================================
    # TOPIC DROPDOWN
    # ========================================================================
    "topic_dropdown": {
        "label": "🏷️ Select the ML topic for this assignment:",
        "topics": [
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
            "PCA / Dimensionality Reduction",
        ],
        "placeholder": "Select a topic...",
        "allow_custom": False,
        "filterable": True,
    },
    
    # ========================================================================
    # ADVANCED OPTIONS
    # ========================================================================
    "advanced_options": {
        "show_rubric_reference": {
            "label": "Show rubric rule references",
            "default": True,
            "help_text": "Include rubric rule IDs (e.g., BP_001) in the feedback"
        },
        "show_timestamps": {
            "label": "Show transcribed text & error analysis (debug)",
            "default": False,
            "help_text": "Display system intermediate outputs for debugging"
        },
    },
    
    # ========================================================================
    # BUTTONS
    # ========================================================================
    "buttons": {
        "submit": {
            "label": "📤 Get Feedback",
            "variant": "primary",
            "size": "lg",
            "icon": "✓",
        },
        "reset": {
            "label": "🔄 Clear Form",
            "variant": "secondary",
            "size": "lg",
            "icon": "↻",
        },
    },
    
    # ========================================================================
    # OUTPUT SECTIONS
    # ========================================================================
    "output_panels": {
        "feedback_card": {
            "title": "📊 Feedback Card",
            "label": "Feedback",
            "format": "markdown",
            "is_visible": True,
            "is_interactive": False,
        },
        "debug_accordion": {
            "title": "🔍 Debug Information (Transcription & Error Analysis)",
            "default_open": False,
            "children": [
                {
                    "name": "transcription",
                    "label": "Transcribed Content",
                    "format": "textbox",
                    "lines": 6,
                    "max_lines": 12,
                },
                {
                    "name": "error_analysis",
                    "label": "Error Analysis (JSON)",
                    "format": "code",
                    "language": "json",
                }
            ]
        }
    },
    
    # ========================================================================
    # TIPS & HELP TEXT
    # ========================================================================
    "help_section": {
        "title": "ℹ️ Tips for Best Results",
        "tips": [
            {
                "icon": "✏️",
                "title": "Clear handwriting",
                "description": "Write legibly with a dark pen or pencil"
            },
            {
                "icon": "📝",
                "title": "Show your work",
                "description": "Include all derivation steps and intermediate results"
            },
            {
                "icon": "🏷️",
                "title": "Label everything",
                "description": "Mark variables, equations, and step numbers clearly"
            },
            {
                "icon": "📐",
                "title": "Include diagrams",
                "description": "Hand-drawn sketches and diagrams are helpful"
            },
            {
                "icon": "📍",
                "title": "Add annotations",
                "description": "Any clarifications or notes you wrote help the system"
            }
        ],
        "processing_time": "Processing time: ~30-60 seconds",
    },
    
    # ========================================================================
    # FOOTER & ABOUT
    # ========================================================================
    "footer": {
        "title": "About MAFA",
        "content": (
            "MAFA combines state-of-the-art AI technologies:\n"
            "- **Qwen2-VL-7B** for multimodal handwritten math transcription\n"
            "- **ChromaDB RAG** for rubric-aligned error detection  \n"
            "- **Fine-tuned Phi-3-mini** for Socratic hint generation\n"
            "- **LangGraph** orchestration for reliable agent workflows"
        ),
        "course_info": {
            "subject": "Machine Learning Foundations",
            "pdfs": 38,
            "units": 4,
            "topics": 18,
            "rubric_rules": 133,
        },
        "disclaimer": (
            "Note: This is a learning assistant. Use feedback to improve your understanding, "
            "not as a replacement for studying course materials."
        ),
        "show_divider": True,
    },
    
    # ========================================================================
    # BEHAVIORAL SETTINGS
    # ========================================================================
    "behavior": {
        "submit_on_enter": False,
        "auto_reset_after_submit": False,
        "enable_keyboard_shortcuts": {
            "submit": "Ctrl+Enter",
            "reset": "Ctrl+R",
        },
        "show_progress_bar": True,
        "show_estimated_time": True,
        "cache_submissions": False,  # Privacy
        "require_confirmation": False,
    },
    
    # ========================================================================
    # VALIDATION RULES
    # ========================================================================
    "validation": {
        "file_required": True,
        "topic_required": True,
        "min_file_size_kb": 10,
        "max_file_size_mb": 50,
        "timeout_seconds": 120,
        "error_messages": {
            "no_file": "❌ Please upload a file before submitting.",
            "no_topic": "❌ Please select a topic from the dropdown.",
            "file_too_large": "❌ File exceeds maximum size of 50 MB.",
            "invalid_format": "❌ File format not supported. Use PDF, JPG, PNG, or WEBP.",
            "processing_timeout": "❌ Processing timed out. Please try again.",
            "processing_error": "❌ An unexpected error occurred. Please try again.",
        }
    },
    
    # ========================================================================
    # ANALYTICS & LOGGING
    # ========================================================================
    "analytics": {
        "enable_logging": True,
        "log_submissions": True,
        "log_errors": True,
        "save_feedback_history": False,  # Privacy
        "metrics_to_track": [
            "submission_count",
            "topics_accessed",
            "avg_processing_time",
            "error_rate",
        ]
    },
}

# ============================================================================
# TOPIC METADATA
# ============================================================================

TOPIC_METADATA = {
    "Decision Trees": {
        "unit": "Unit 1",
        "difficulty": "Beginner",
        "concepts": ["Information Gain", "Entropy", "Splitting Criteria"],
    },
    "Backpropagation": {
        "unit": "Unit 2",
        "difficulty": "Advanced",
        "concepts": ["Chain Rule", "Gradient Descent", "Weight Updates"],
    },
    "Forward Propagation": {
        "unit": "Unit 2",
        "difficulty": "Intermediate",
        "concepts": ["Activation Functions", "Layer Computation"],
    },
    "SVM - Hard Margin": {
        "unit": "Unit 3",
        "difficulty": "Advanced",
        "concepts": ["Hyperplane", "Margin Maximization", "Support Vectors"],
    },
    "SVM - Soft Margin": {
        "unit": "Unit 3",
        "difficulty": "Advanced",
        "concepts": ["Slack Variables", "Regularization", "Non-separable Data"],
    },
    "Naive Bayes": {
        "unit": "Unit 1",
        "difficulty": "Beginner",
        "concepts": ["Conditional Probability", "Independence Assumption", "Laplace Smoothing"],
    },
    "HMM - Likelihood Problem": {
        "unit": "Unit 4",
        "difficulty": "Advanced",
        "concepts": ["Forward Algorithm", "Dynamic Programming", "Probability Computation"],
    },
    "HMM - Decoding Problem": {
        "unit": "Unit 4",
        "difficulty": "Advanced",
        "concepts": ["Viterbi Algorithm", "Hidden States", "Most Likely Path"],
    },
    "HMM - Learning Problem": {
        "unit": "Unit 4",
        "difficulty": "Advanced",
        "concepts": ["Baum-Welch", "EM Algorithm", "Parameter Estimation"],
    },
    "EM Algorithm": {
        "unit": "Unit 4",
        "difficulty": "Advanced",
        "concepts": ["Expectation Step", "Maximization Step", "Latent Variables"],
    },
    "K-Means Clustering": {
        "unit": "Unit 3",
        "difficulty": "Intermediate",
        "concepts": ["Centroid", "Cluster Assignment", "Convergence"],
    },
    "CNN Trainable Parameters": {
        "unit": "Unit 2",
        "difficulty": "Intermediate",
        "concepts": ["Convolutional Filters", "Pooling", "Parameter Count"],
    },
    "Random Forest": {
        "unit": "Unit 1",
        "difficulty": "Intermediate",
        "concepts": ["Ensemble Methods", "Bootstrap Aggregating", "Feature Randomness"],
    },
    "Bagging & Boosting": {
        "unit": "Unit 1",
        "difficulty": "Intermediate",
        "concepts": ["Variance Reduction", "AdaBoost", "Sequential Learning"],
    },
    "Gradient Boosting": {
        "unit": "Unit 1",
        "difficulty": "Advanced",
        "concepts": ["Residual Learning", "Learning Rate", "Tree Depth"],
    },
    "Logistic Regression": {
        "unit": "Unit 1",
        "difficulty": "Beginner",
        "concepts": ["Sigmoid Function", "Cross Entropy", "Binary Classification"],
    },
    "Performance Metrics": {
        "unit": "Unit 1",
        "difficulty": "Beginner",
        "concepts": ["Precision", "Recall", "F1-Score", "AUC-ROC"],
    },
    "PCA / Dimensionality Reduction": {
        "unit": "Unit 3",
        "difficulty": "Intermediate",
        "concepts": ["Eigenvalues", "Covariance Matrix", "Variance Explained"],
    },
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_topic_metadata(topic: str) -> Dict[str, Any]:
    """Get metadata for a specific topic."""
    return TOPIC_METADATA.get(topic, {})

def get_all_topics() -> List[str]:
    """Get list of all topics."""
    return UI_CONFIG["topic_dropdown"]["topics"]

def save_config(output_path: str = "config/ui_config.json") -> None:
    """Save configuration to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(UI_CONFIG, f, indent=2)
    print(f"✓ Configuration saved to {output_path}")

def load_config(config_path: str = "config/ui_config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

# ============================================================================
# QUICK START
# ============================================================================

if __name__ == "__main__":
    print("MAFA UI Configuration")
    print("=" * 70)
    print(f"\nApp Name: {UI_CONFIG['app_full_name']}")
    print(f"Version: {UI_CONFIG['version']}")
    print(f"Topics: {len(UI_CONFIG['topic_dropdown']['topics'])}")
    print(f"Theme: {UI_CONFIG['theme']['primary_color']}")
    
    print("\nAvailable Topics:")
    for i, topic in enumerate(get_all_topics(), 1):
        metadata = get_topic_metadata(topic)
        unit = metadata.get('unit', 'N/A')
        difficulty = metadata.get('difficulty', 'N/A')
        print(f"  {i:2}. {topic:.<40} {unit:.<12} {difficulty}")
    
    print("\n" + "=" * 70)
    print("\nTo use in app.py:")
    print("""
    from config.ui_config import UI_CONFIG, get_all_topics
    
    # Get all topics
    topics = get_all_topics()
    
    # Access specific config
    theme = UI_CONFIG['theme']
    file_config = UI_CONFIG['file_upload']
    """)
