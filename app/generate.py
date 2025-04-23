from __future__ import annotations

import io
import os
import re
from typing import List, Dict, Any, Tuple
import json

import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import pdfplumber
from mistralai import Mistral
import ollama

# ──────────────────────────── global singletons ────────────────────────────
_MODEL_NAME = "intfloat/e5-small-v2"
_tokenizer = None
_model = None

# ────────────────────────────── helpers ─────────────────────────────────────

def _get_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    if _model is None:
        _model = AutoModel.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model

def _encode_text(texts: List[str]) -> torch.Tensor:
    tokenizer, model = _get_model()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

def _extract_text_from_pdf(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)

def _extract_text_from_docx(data: bytes) -> str:
    try:
        import docx  # type: ignore  # python-docx
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "python-docx is not installed (or wrong 'docx' package present).\n"
            "Run:  pip uninstall docx  &&  pip install python-docx"
        ) from e

    with io.BytesIO(data) as buffer:
        document = docx.Document(buffer)
        paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)

def _chunk_text(text: str, max_chars: int = 350) -> List[str]:
    chunks: List[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > max_chars:
            split_at = paragraph.rfind(" ", 0, max_chars)
            if split_at == -1:
                split_at = max_chars
            chunks.append(paragraph[:split_at].strip())
            paragraph = paragraph[split_at:].strip()
        if paragraph:
            chunks.append(paragraph)
    return chunks

def _build_vector_index(chunks: List[str]) -> Tuple[faiss.IndexFlatIP, List[str]]:
    vectors = _encode_text(chunks).numpy()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks

def _search_top_k(index: faiss.IndexFlatIP, chunks: List[str], query: str, k: int) -> List[str]:
    q_vec = _encode_text([query]).numpy()
    scores, ids = index.search(q_vec, k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]

def _build_prompt(jd: str, retrieved: List[str], extra: str) -> List[Dict[str, str]]:
    system = (
        "You are an expert career assistant who crafts professional resumes and cover letters. Format the output as follows:\n\n"
        "RESUME FORMAT:\n"
        "- Organize into sections: SUMMARY, SKILLS \n"
        "- write summary and skills section based on the the jobs description\n"
        "- Keep formatting clean and professional\n\n"
        "COVER LETTER FORMAT:\n"
        "- Start with your contact info and date\n"
        "- Include recipient's name/title/company if provided\n" 
        "- why me? answer this with 3-4 focused paragraphs highlighting relevant experience\n"
        "- why [company]? answer with company related info and the users interest in the role\n\n"
        "Output must be JSON with two keys:\n"
        "- resume_bullets: The full formatted resume text\n"
        "- cover_letter: The full formatted cover letter text"
    )

    user_content = {
        "job_description": jd,
        "relevant_resume_fragments": retrieved,
        "extra_notes": extra,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": str(user_content)},
    ]

def _call_mistral(messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY env var not set")

    client = Mistral(api_key=api_key)
    resp = client.chat.complete(model=model, messages=messages)
    txt = resp.choices[0].message.content.strip()

    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        raise ValueError(f"Mistral response was not valid JSON:\n{txt}") from e

def _call_ollama(messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    try:
        # Check if model exists, but don't fail if we can't check
        try:
            response = ollama.list()
            if isinstance(response, dict) and 'models' in response:
                available_models = [model_info.get('name', '') for model_info in response['models']]
                if model not in available_models:
                    print(f"Warning: Model '{model}' may need to be pulled first")
        except Exception as e:
            print(f"Warning: Could not check for model availability: {str(e)}")
            
        # Proceed with chat regardless
        response = ollama.chat(model=model, messages=messages)
        
        # Handle the new response format and extract content
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            txt = response.message.content.strip()
        elif isinstance(response, dict):
            if 'message' in response and isinstance(response['message'], dict):
                txt = response['message'].get('content', '').strip()
            else:
                txt = str(response).strip()
        else:
            txt = str(response).strip()
            
        # Try to extract JSON from the response if it contains thinking process
        try:
            # Look for JSON-like structure in the text
            json_start = txt.find('{')
            json_end = txt.rfind('}')
            if json_start >= 0 and json_end > json_start:
                potential_json = txt[json_start:json_end + 1]
                return json.loads(potential_json)
            else:
                # If no JSON-like structure found, try parsing the whole text
                return json.loads(txt)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not extract valid JSON from Ollama response:\n{txt}") from e
            
    except Exception as e:
        if "no such file" in str(e).lower():
            raise RuntimeError(f"Model '{model}' not found. Try running: ollama pull {model}") from e
        raise RuntimeError(f"Ollama API error: {str(e)}") from e

# ──────────────────────────── public API ───────────────────────────────────

def generate_documents(
    *,
    resume_bytes: bytes,
    filename: str,
    job_desc: str,
    extra_notes: str = "",
    model: str = "mistral-small",
    provider: str = "mistral",
    top_k: int = 8,
) -> Dict[str, str]:
    """Main entry called by the Streamlit app."""

    # 1. extract
    if filename.lower().endswith(".pdf"):
        resume_text = _extract_text_from_pdf(resume_bytes)
    elif filename.lower().endswith(".docx"):
        resume_text = _extract_text_from_docx(resume_bytes)
    else:
        raise ValueError("Unsupported résumé format. Upload PDF or DOCX.")

    if not resume_text.strip():
        raise ValueError("Could not extract text from the résumé.")

    # 2. chunk & embed index
    chunks = _chunk_text(resume_text)
    index, store = _build_vector_index(chunks)

    # 3. retrieve top‑k chunks vs job_desc
    retrieved = _search_top_k(index, store, job_desc, top_k)

    # 4. call selected model
    messages = _build_prompt(job_desc, retrieved, extra_notes)
    if provider == "mistral":
        llm_json = _call_mistral(messages, model=model)
    else:  # ollama
        llm_json = _call_ollama(messages, model=model)

    return {
        "resume_txt": llm_json.get("resume_bullets", ""),
        "cover_letter_txt": llm_json.get("cover_letter", "")
    }
