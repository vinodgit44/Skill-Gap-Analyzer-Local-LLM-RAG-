import os
import torch
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# =======================================================
# AUTO-DETECT DEVICE (future GPU supported)
# =======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Running on: **{device.upper()}**")


# =======================================================
# Streamlit UI
# =======================================================
st.set_page_config(page_title="JD–Resume Analyzer", layout="wide")
st.title("📄 Resume + JD Skill Gap Analyzer (Local LLM – GPU/CPU)")


resume_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])
jd_file     = st.file_uploader("Upload Job Description (.txt)", type=["txt"])


if st.button("🚀 Run Analysis"):

    if resume_file is None or jd_file is None:
        st.error("Upload BOTH files.")
        st.stop()

    # Save uploaded text
    os.makedirs("tmp", exist_ok=True)
    resume_path = "tmp/resume.txt"
    jd_path     = "tmp/jd.txt"

    open(resume_path, "w").write(resume_file.read().decode())
    open(jd_path, "w").write(jd_file.read().decode())


    # =======================================================
    # Load Docs
    # =======================================================
    resume_docs = TextLoader(resume_path).load()
    jd_docs     = TextLoader(jd_path).load()

    for d in resume_docs: d.metadata["source"] = "resume"
    for d in jd_docs:     d.metadata["source"] = "jd"

    all_docs = resume_docs + jd_docs


    # =======================================================
    # Embeddings (GPU or CPU)
    # =======================================================
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    if os.path.exists("chroma_store"):
        vector_store = Chroma(
            persist_directory="chroma_store",
            embedding_function=embeddings
        )
    else:
        vector_store = Chroma.from_documents(
            all_docs,
            embedding=embeddings,
            persist_directory="chroma_store"
        )
        vector_store.persist()

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})


    # =======================================================
    # LOAD LOCAL LLM (GPU OR CPU)
    # =======================================================
    st.write(f"Loading TinyLlama on {device}…")

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = Aut
