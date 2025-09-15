from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
from typing import List, Dict, Callable
import os
import pickle

INDEX_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(INDEX_DIR, exist_ok=True)
_INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
_META_FILE = os.path.join(INDEX_DIR, "meta.pkl")

def clean_text(s: str) -> str:
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(page_record: Dict, chunk_size_chars: int = 1200, overlap_chars: int = 200) -> List[Dict]:
    text = clean_text(page_record["text"])
    chunks = []
    start = 0
    L = len(text)
    if L == 0:
        return []
    while start < L:
        end = min(start + chunk_size_chars, L)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "doc": page_record["doc"],
                "page": page_record["page"],
                "text": chunk_text
            })
        start = end - overlap_chars
        if start < 0: start = 0
        if start >= L:
            break
    return chunks

def save_index(index, metadatas):
    with open(_META_FILE, "wb") as f:
        pickle.dump(metadatas, f)
    faiss.write_index(index, _INDEX_FILE)

def load_index():
    if os.path.exists(_INDEX_FILE) and os.path.exists(_META_FILE):
        idx = faiss.read_index(_INDEX_FILE)
        with open(_META_FILE, "rb") as f:
            meta = pickle.load(f)
        return {"index": idx, "metadatas": meta, "embedder_model_name": "all-MiniLM-L6-v2"}
    return None

def build_faiss_index(pages: List[Dict], chunk_size=1200, overlap=200, embedder_model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64, progress_callback: Callable[[float,str],None]=None):
    """
    Build FAISS index from list of page records in batches.
    progress_callback(progress_fraction, message) -> called with fraction [0..1]
    """
    # chunk pages
    chunks = []
    for p in pages:
        ch = chunk_text(p, chunk_size_chars=chunk_size, overlap_chars=overlap)
        chunks.extend(ch)
    if len(chunks) == 0:
        raise ValueError("No text extracted from PDFs.")
    if progress_callback:
        progress_callback(0.01, f"Prepared {len(chunks)} chunks")

    # load model
    model = SentenceTransformer(embedder_model_name)

    texts = [c["text"] for c in chunks]
    n = len(texts)
    all_embeddings = []
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i+batch_size]
        emb = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        all_embeddings.append(emb)
        if progress_callback:
            progress_callback(0.01 + 0.7 * (i+len(batch_texts)) / n, f"Embedding batch {i//batch_size + 1}/{(n+batch_size-1)//batch_size}")

    embeddings = np.vstack(all_embeddings)
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    if progress_callback:
        progress_callback(0.95, "Built FAISS index")
    # persist
    save_index(index, chunks)
    if progress_callback:
        progress_callback(1.0, "Saved index to disk")
    return {"index": index, "metadatas": chunks, "embedder_model_name": embedder_model_name}

def query_faiss(index, metadatas, query: str, top_k: int = 5, embedder_model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(embedder_model_name)
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta = metadatas[idx]
        results.append({
            "score": float(score),
            "doc": meta["doc"],
            "page": meta["page"],
            "text": meta["text"]
        })
    return results
