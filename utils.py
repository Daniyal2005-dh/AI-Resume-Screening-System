import os
from pathlib import Path
from typing import List, Dict

# PDF text extraction
from pdfminer.high_level import extract_text

# NLP utilities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF file.

    Args:
        pdf_path (Path): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        text = extract_text(str(pdf_path))
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {pdf_path}: {e}")


def preprocess_text(text: str) -> str:
    """Basic preprocessing for resume text.

    - Lowercase
    - Remove non‑alphanumeric characters (keep spaces)
    - Collapse multiple spaces
    """
    # Lowercase
    text = text.lower()
    # Remove non‑alphanumeric (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rank_candidates(job_description: str, candidates: List[Dict[str, str]]) -> List[Dict[str, any]]:
    """Rank candidates based on TF‑IDF cosine similarity to the job description.

    Args:
        job_description (str): The full job description text.
        candidates (List[Dict]): List of dicts with keys ``name`` and ``text`` (pre‑processed).

    Returns:
        List[Dict]: Candidates sorted by descending similarity. Each dict contains ``name``, ``score``.
    """
    # Prepare corpus: job description first, then each candidate's text
    corpus = [job_description] + [c["text"] for c in candidates]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    job_vec = tfidf_matrix[0]
    candidate_vecs = tfidf_matrix[1:]
    # Compute cosine similarities
    similarities = cosine_similarity(job_vec, candidate_vecs).flatten()
    # Attach scores to candidates
    ranked = []
    for cand, score in zip(candidates, similarities):
        ranked.append({"name": cand["name"], "score": float(score)})
    # Sort descending by score
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
