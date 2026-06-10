import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from utils import extract_text_from_pdf, preprocess_text, rank_candidates

st.set_page_config(page_title="AI Resume Screening", layout="wide")

# --- Custom CSS for premium glassmorphism UI ---
custom_css = """
/* Glassmorphism container */
.glass {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 2rem;
    margin-top: 1rem;
    box-shadow: 0 4px 30px rgba(0,0,0,0.1);
}
/* Smooth hover animation */
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 20px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
}
.card {
    background: rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.25);
    transition: all 0.3s ease;
}
"""
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

st.title("🤖 AI Resume Screening System")
st.subheader("Upload candidate resumes and rank them against a job description")

with st.container():
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload candidate resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files containing candidate resumes."
    )
    job_description = st.text_area(
        "Job Description",
        height=150,
        placeholder="Paste the full job description here...",
        help="The system will rank candidates based on similarity to this description."
    )
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_files and job_description:
    st.info("Processing resumes… this may take a few seconds.")
    candidates = []
    for uploaded in uploaded_files:
        # Save to a temporary location
        temp_path = Path("tmp") / uploaded.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        text = extract_text_from_pdf(temp_path)
        cleaned = preprocess_text(text)
        candidates.append({"name": uploaded.name, "text": cleaned})
        # Cleanup file after extraction
        temp_path.unlink(missing_ok=True)
    # Rank candidates
    rankings = rank_candidates(job_description, candidates)
    # Display results
    st.success("✅ Ranking completed!")
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    for rank, info in enumerate(rankings, start=1):
        st.markdown(
            f"""
            <div class='card'>
                <h4>#{rank} – {info['name']}</h4>
                <p>Similarity: <b>{info['score']:.2%}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("Please upload at least one resume and provide a job description to see rankings.")
