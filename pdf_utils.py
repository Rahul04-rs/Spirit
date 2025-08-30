# backend/pdf_utils.py

import io
import re
from pypdf import PdfReader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from backend.utils import clean_text
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

def extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF, with OCR fallback if needed."""
    text = ""

    # Try text extraction via PyPDF
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    except Exception as e:
        print(f"[ERROR] PyPDF extraction failed: {e}")

    # Fallback to OCR if no extractable text
    if not text.strip():
        print("[INFO] No text found, using OCR fallback...")
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text += pytesseract.image_to_string(img) + "\n"

    return text.strip()


def extract_text_pages(file_path):
    """
    Split extracted text into 'pages'.
    If actual page separation not available, split by form feed or fallback to one big string.
    """
    raw_text = extract_pdf_text(file_path)
    pages = raw_text.split("\f") if "\f" in raw_text else [raw_text]
    return [clean_text(p) for p in pages]


def pages_to_chunks(pages):
    """
    Turn list of page texts into overlapping chunks using CHUNK_SIZE and CHUNK_OVERLAP.
    Adds 'meta': {'page': page_number} to each chunk.
    """
    chunks = []
    for page_num, page_text in enumerate(pages, start=1):
        words = page_text.split()
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "meta": {"page": page_num}
                })
    return chunks
