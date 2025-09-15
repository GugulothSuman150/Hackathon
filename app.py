import streamlit as st
from utils.pdf_utils import extract_text_from_pdfs, save_uploaded_file
from utils.embed_utils import build_faiss_index, query_faiss, load_index
from utils.llm_utils import generate_answer
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="StudyMate", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style> .reportview-container { background: #f8fafc;} </style>", unsafe_allow_html=True)

st.title("📚 StudyMate — PDF Q&A (Local + OCR.space)")

with st.sidebar:
    st.header("Options")
    st.markdown("**Answer mode**")
    llm_choice = st.selectbox("Answer mode", ["retrieval-only", "flan-t5-small (local)"], index=0)
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 5)
    chunk_size = st.slider("Chunk size (chars)", 600, 2000, 1200, step=100)
    overlap = st.slider("Chunk overlap (chars)", 50, 400, 200, step=10)
    st.markdown("---")
    st.markdown("Built for local demo; OCR.space used for scanned PDFs if needed.")
    st.markdown("ⓘ OCR API key is taken from .env OCR_SPACE_API_KEY")

st.subheader("1) Upload PDF(s)")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# Try to load existing index on disk
if "index_loaded" not in st.session_state:
    idx_data = load_index()
    if idx_data:
        st.session_state["index"] = idx_data["index"]
        st.session_state["metadatas"] = idx_data["metadatas"]
        st.session_state["embedder_model_name"] = idx_data["embedder_model_name"]
        st.session_state["index_loaded"] = True

if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) selected.")
    if st.button("Process & Index PDFs"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        def progress_cb(fraction, message):
            try:
                progress_bar.progress(min(1.0, max(0.0, fraction)))
                status_text.text(message)
            except Exception:
                pass
        with st.spinner("Extracting text and building FAISS index..."):
            docs = extract_text_from_pdfs(uploaded_files, show_progress_callback=progress_cb)
            index_data = build_faiss_index(docs, chunk_size=chunk_size, overlap=overlap, progress_callback=progress_cb)
            st.session_state["index"] = index_data["index"]
            st.session_state["metadatas"] = index_data["metadatas"]
            st.session_state["embedder_model_name"] = index_data["embedder_model_name"]
            st.success(f"Indexed {len(index_data['metadatas'])} chunks from {len(uploaded_files)} PDF(s).")
            progress_bar.empty()
            status_text.empty()

if "index" in st.session_state:
    st.subheader("2) Ask a question")
    query = st.text_input("Type your question here and press Enter (or click Ask)")
    ask_btn = st.button("Ask")
    if ask_btn or (query and st.session_state.get("auto_ask", False)):
        if not query:
            st.warning("Please type a question.")
        else:
            with st.spinner("Retrieving relevant chunks..."):
                retrieved = query_faiss(
                    st.session_state["index"],
                    st.session_state["metadatas"],
                    query,
                    top_k=top_k,
                    embedder_model_name=st.session_state.get("embedder_model_name", "all-MiniLM-L6-v2")
                )
            st.markdown("### 🔎 Retrieved snippets")
            for i, r in enumerate(retrieved):
                st.markdown(f"**Source {i+1}:** {r['doc']} — page {r['page']} — score {r['score']:.3f}")
                st.write(r['text'][:800] + ("..." if len(r['text'])>800 else ""))
                if st.button(f"Show full source {i+1}"):
                    st.code(r['text'][:4000])

            with st.spinner("Generating answer..."):
                if llm_choice.startswith("flan") and llm_choice:
                    answer = generate_answer(query, retrieved, mode="flan")
                else:
                    answer = generate_answer(query, retrieved, mode="retrieval")
            st.markdown("### ✅ Answer")
            st.info(answer)
            st.markdown("### 📚 Sources")
            for i, r in enumerate(retrieved):
                st.write(f"- [Source {i+1}] {r['doc']} — page {r['page']} (score {r['score']:.3f})")
else:
    st.info("Upload and process PDFs to enable Q&A.")
