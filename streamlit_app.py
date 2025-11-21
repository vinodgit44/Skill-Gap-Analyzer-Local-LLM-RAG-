import os
import torch
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# =======================================================
# FORCE CPU
# =======================================================
device = "cpu"
st.sidebar.write("Using **CPU mode** (GTX 1060 unsupported for CUDA 12)")


# =======================================================
# Streamlit UI
# =======================================================
st.set_page_config(page_title="JD–Resume Analyzer", layout="wide")
st.title("📄 Resume + JD Skill Gap Analyzer (Local LLM – CPU)")


resume_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])
jd_file     = st.file_uploader("Upload JD (.txt)", type=["txt"])


if st.button("🚀 Run Analysis"):

    if resume_file is None or jd_file is None:
        st.error("Upload both files.")
        st.stop()


    # Save files
    os.makedirs("tmp", exist_ok=True)
    resume_path = "tmp/resume.txt"
    jd_path     = "tmp/jd.txt"

    open(resume_path, "w").write(resume_file.read().decode())
    open(jd_path, "w").write(jd_file.read().decode())


    # Load docs
    resume_docs = TextLoader(resume_path).load()
    jd_docs     = TextLoader(jd_path).load()

    all_docs = []
    for d in resume_docs:
        d.metadata["source"] = "resume"
        all_docs.append(d)

    for d in jd_docs:
        d.metadata["source"] = "jd"
        all_docs.append(d)


    # EMBEDDINGS (CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Chroma
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


    # LOAD TINYLLAMA (CPU)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.7,
        device=-1  # CPU
    )

    def LLM(prompt):
        return pipe(prompt)[0]["generated_text"]


    # RAG Function (FIXED for LangChain v1.x)
    def RAG(query):
        docs = retriever.invoke(query)  # FIXED HERE ✅
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Use ONLY this context to answer:

Context:
{context}

Question: {query}

Answer:
"""
        return LLM(prompt)


    # RESULTS
    st.subheader("📌 JD Skills")
    st.write(RAG("Extract all skills from the job description."))

    st.subheader("📌 Resume Skills")
    st.write(RAG("Extract all skills from the resume."))

    st.subheader("📊 Skill Gap Analysis")
    st.write(RAG("Compare the resume and JD and list skill gaps."))

    st.subheader("🎯 Learning Plan")
    st.write(RAG("Give a learning roadmap and 5 project ideas."))

    st.subheader("🧮 Job Fit Score")
    st.write(RAG("Give a job-fit score out of 100 with bullet points."))

    st.success("Analysis Complete!")
else:
    st.info("Upload both files and click **Run Analysis**.")
