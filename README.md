# AnswerForce — PDF Q&A (Local Streamlit Version)

This project is a local, Streamlit-based implementation of AnswerForce:
an AI-powered PDF question-answering system optimized for hackathon demos.
It uses local (free) models where possible and requires no cloud API keys.

## Features
- Upload multiple PDFs
- Extract and chunk text (PyMuPDF)
- Build embeddings with SentenceTransformers + FAISS
- Retrieval of top-k relevant chunks
- Local answer generation using `google/flan-t5-small` (HuggingFace) or retrieval-only mode
- Simple, styled Streamlit UI with sidebar and source citations

## Requirements
Install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
