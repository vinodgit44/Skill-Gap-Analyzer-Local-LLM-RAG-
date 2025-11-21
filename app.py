import os
import torch

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ====================================================
# FORCE CPU
# ====================================================
device = "cpu"
print("Using:", device)


# ====================================================
# PATHS
# ====================================================
RESUME_PATH = "sample_data/sample_resume.txt"
JD_PATH     = "sample_data/sample_jd.txt"
CHROMA_DIR  = "chroma_store"


# ====================================================
# LOAD DOCUMENTS
# ====================================================
resume_docs = TextLoader(RESUME_PATH).load()
jd_docs     = TextLoader(JD_PATH).load()

for d in resume_docs:
    d.metadata["source"] = "resume"

for d in jd_docs:
    d.metadata["source"] = "job_description"

all_docs = resume_docs + jd_docs


# ====================================================
# EMBEDDINGS (CPU)
# ====================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

if os.path.exists(CHROMA_DIR):
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
else:
    vector_store = Chroma.from_documents(
        all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vector_store.persist()

retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# ====================================================
# LOAD LOCAL TINYLLAMA MODEL (CPU)
# ====================================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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


# ====================================================
# MANUAL RAG PIPELINE (LANGCHAIN v1.x COMPATIBLE)
# ====================================================
def RAG(query):
    docs = retriever.invoke(query)   # FIXED HERE ✅
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use ONLY this context to answer:

Context:
{context}

Question: {query}

Answer:
"""
    return LLM(prompt)


# ====================================================
# RUN ANALYSIS
# ====================================================
print(RAG("Extract all skills from the job description."))
print(RAG("Extract all skills from the resume."))
print(RAG("Find skill gaps between resume and JD."))
print(RAG("Give a learning roadmap with project ideas."))
print(RAG("Give a job-fit score out of 100 with bullet points."))
