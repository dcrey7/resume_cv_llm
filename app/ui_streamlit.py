"""ui_streamlit.py

Single‑file Streamlit front‑end for the Résumé + Cover‑Letter Agent.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
import streamlit as st
from generate import generate_documents
from pathlib import Path
import datetime

# Load environment variables from .env file
load_dotenv()

# ────────────────────────── config ──────────────────────────
st.set_page_config(page_title="Resume Assistant", layout="wide")

# Default Ollama models that are installed locally
OLLAMA_MODELS = ["deepseek-llm", "huihui_ai/deepseek-r1-abliterated:8b"]

# ─────────────────────── session state ───────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

# ──────────────────────── sidebar ──────────────────────────────
with st.sidebar:
    st.title("📄 Resume Assistant")
    
    resume_file = st.file_uploader("Upload your résumé (PDF or DOCX)", type=["pdf", "docx"])
    
    job_desc = st.text_area(
        "Paste the job description",
        height=300,
        help="The full job posting you're applying to",
    )
    
    extra_notes = st.text_area(
        "Additional notes (optional)",
        help="Any special instructions or emphasis",
    )

    # Model selection
    provider = st.radio("AI Provider", ["mistral", "ollama"])
    
    if provider == "mistral":
        API_KEY = os.getenv("MISTRAL_API_KEY")
        if not API_KEY:
            st.error(
                "Environment variable **MISTRAL_API_KEY** not found.\n"
                "Export it before launching the app (e.g. `export MISTRAL_API_KEY=sk‑live‑…`)."
            )
            st.stop()
        model = st.selectbox(
            "Model",
            ["mistral-small", "mistral-medium", "mistral-large"],
            index=0,
        )
    else:  # ollama
        model = st.selectbox(
            "Model",
            OLLAMA_MODELS,
            index=0,
            help="Make sure to pull the model first using 'ollama pull model-name'"
        )
    
    generate_click = st.button("✨ Generate", type="primary")

# ─────────────────────── main panel ────────────────────────────
if resume_file and job_desc and generate_click:
    resume_bytes = resume_file.read()
    
    with st.spinner("🤖 Working on it..."):
        try:
            result = generate_documents(
                resume_bytes=resume_bytes,
                filename=resume_file.name,
                job_desc=job_desc,
                extra_notes=extra_notes,
                model=model,
                provider=provider,
            )
            st.session_state.result = result
        except Exception as e:
            if "not found, try pulling it first" in str(e):
                st.error(f"Error: Model '{model}' is not available. Please run 'ollama pull {model}' in your terminal first.")
            else:
                st.error(f"Error: {str(e)}")
            st.stop()

if st.session_state.result:
    result = st.session_state.result
    
    # Style and format for professional documents
    st.markdown("""
        <style>
        .document {
            background-color: white;
            padding: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            font-family: 'Times New Roman', serif;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .section {
            margin-bottom: 15px;
        }
        .section-title {
            font-weight: bold;
            border-bottom: 1px solid #ddd;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Résumé")
        with st.container():
            st.markdown(f'<div class="document">{result["resume_txt"]}</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("✉️ Cover Letter")
        with st.container():
            st.markdown(f'<div class="document">{result["cover_letter_txt"]}</div>', unsafe_allow_html=True)

# ───────────────────────── footer ───────────────────────────────
with st.expander("ℹ️ About this app"):
    st.markdown(
        "This app helps you create tailored résumé bullets and a cover letter "
        f"using either Mistral AI API or local Ollama models. Currently using: {provider} - {model}"
    )
