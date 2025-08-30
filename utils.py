# backend/utils.py
import re

def clean_text(text: str) -> str:
    """Clean text: remove extra spaces, newlines, non-breaking spaces."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
