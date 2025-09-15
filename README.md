# StudyMate — PDF Q&A (Local Streamlit Version)

This project is a local, Streamlit-based implementation of StudyMate:
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
```

> Note: The first run will download models from Hugging Face (embedding model and optionally `flan-t5-small`).
> These downloads require internet and may take time.

## Run
```bash
streamlit run app.py
```

## Notes
- This is designed for **local** demos. For larger documents or better generation quality, consider using GPU or a cloud LLM (e.g., IBM Watsonx or hosted HF inference).
- If you don't want generation, choose **retrieval-only** mode for deterministic, source-grounded responses.

## Project structure
- `app.py` — Streamlit app
- `utils/pdf_utils.py` — PDF extraction helpers
- `utils/embed_utils.py` — chunking, embedding, FAISS index
- `utils/llm_utils.py` — prompt building & local generation
