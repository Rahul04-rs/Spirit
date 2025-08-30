import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from backend.pdf_utils import extract_text_pages, pages_to_chunks
from backend.embed_index import FaissRetriever
from backend.rag import compose_prompt, generate_answer
from backend.retriever import Retriever as FaissRetriever

st.set_page_config(page_title="StudyMate", layout="wide")
st.title("📚 StudyMate - AI PDF Q&A System")

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

retriever = FaissRetriever()

if uploaded_files:
    all_chunks = []
    os.makedirs("demo_pdfs", exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join("demo_pdfs", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        pages = extract_text_pages(file_path)
        chunks = pages_to_chunks(pages)
        all_chunks.extend(chunks)

    retriever.add_chunks(all_chunks)
    st.success(f"Processed {len(all_chunks)} chunks from {len(uploaded_files)} PDF(s).")

query = st.text_input("Ask a question about your uploaded PDFs:")

if query:
    if not retriever.chunks:
        st.warning("Please upload PDFs first.")
    else:
        retrieved_chunks = retriever.retrieve(query, top_k=5)
        prompt = compose_prompt(query, retrieved_chunks)
        answer = generate_answer(prompt)
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Sources:")
        for i, c in enumerate(retrieved_chunks):
            st.write(f"Source {i+1} (Page {c['meta']['page']}): {c['text'][:200]}...")
