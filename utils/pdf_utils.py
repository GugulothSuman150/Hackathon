import fitz
import tempfile
import os
from typing import List, Dict
import streamlit as st
import requests
import io

OCR_API_KEY = os.getenv("OCR_SPACE_API_KEY", "").strip()

def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit uploaded file to a temp path and return path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.getbuffer())
    tf.flush()
    tf.close()
    return tf.name

def _is_scanned_text(text: str, threshold: int = 40) -> bool:
    """Detect if a page is likely scanned (very short extracted text)."""
    if not text:
        return True
    if len(text.strip()) < threshold:
        return True
    # if text has very few alphabets, treat as scanned
    letters = sum(c.isalpha() for c in text)
    if letters / max(1, len(text)) < 0.2:
        return True
    return False

def _ocr_space_image_bytes(image_bytes: bytes, api_key: str) -> str:
    """Call OCR.space with an image bytes and return extracted text (or empty string)."""
    if not api_key:
        return ""
    try:
        url = "https://api.ocr.space/parse/image"
        files = { "file": ("page.png", image_bytes, "image/png") }
        data = { "apikey": api_key, "language": "eng", "isOverlayRequired": False }
        resp = requests.post(url, files=files, data=data, timeout=120)
        resp.raise_for_status()
        j = resp.json()
        if j.get("IsErroredOnProcessing"):
            return ""
        parsed = j.get("ParsedResults", [])
        if not parsed:
            return ""
        text = parsed[0].get("ParsedText", "") or ""
        return text
    except Exception as e:
        # Do not crash the whole pipeline for OCR errors
        return ""

def extract_text_from_pdfs(uploaded_files, show_progress_callback=None) -> List[Dict]:
    """
    Accepts a list of Streamlit uploaded file objects.
    Returns list of page records: {'doc': filename, 'page': page_num, 'text': text}
    Strategy:
      - For each page: try fast text extraction via PyMuPDF.
      - If text seems empty or scanned-like, render page to PNG and call OCR.space per page.
    show_progress_callback(progress_fraction, message) -> optional callback for UI feedback
    """
    pages = []
    total_files = len(uploaded_files)
    file_idx = 0
    for uf in uploaded_files:
        file_idx += 1
        path = save_uploaded_file(uf)
        doc = fitz.open(path)
        total_pages = len(doc)
        for i in range(total_pages):
            page = doc[i]
            text = page.get_text("text") or ""
            if _is_scanned_text(text):
                # render to PNG and call OCR.space (per-page) to avoid large uploads
                try:
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    ocr_text = _ocr_space_image_bytes(img_bytes, OCR_API_KEY)
                    if ocr_text and len(ocr_text.strip()) > 10:
                        text = ocr_text
                except Exception as e:
                    # fallback to whatever text we had
                    pass
            pages.append({
                "doc": os.path.basename(path),
                "page": i+1,
                "text": text if text else ""
            })
            # progress callback update per page
            if show_progress_callback:
                frac = ((file_idx-1) + (i+1)/max(1,total_pages)) / max(1,total_files)
                try:
                    show_progress_callback(min(1.0, frac), f"Processing {os.path.basename(path)} page {i+1}/{total_pages}")
                except Exception:
                    pass
        doc.close()
    return pages
