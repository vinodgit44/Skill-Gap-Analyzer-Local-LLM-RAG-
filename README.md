



# 📄 **README.md — JD–Resume Skill Gap Analyzer (Local LLM + RAG)**

<p align="center">
  <img src="assets/banner.png" alt="JD–Resume Analyzer Banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/LangChain-1.x-orange" />
  <img src="https://img.shields.io/badge/ChromaDB-Local%20Vector%20DB-green" />
  <img src="https://img.shields.io/badge/LLM-TinyLlama%20%2F%20Qwen%20%2F%20GPT2-red" />
  <img src="https://img.shields.io/badge/UI-Streamlit-ff69b4" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

---

# 🚀 **JD–Resume Skill Gap Analyzer (Local LLM + RAG)**

A privacy-preserving **AI career analysis tool** that compares your **resume** with any **job description** using a **fully local RAG pipeline** powered by:

* **LangChain v1.x**
* **ChromaDB**
* **MiniLM Embeddings**
* **Local LLMs (TinyLlama / GPT-2 / Qwen)**
* **Streamlit UI**

> ⚡ Works with **Local CPU**, **Local GPU (future)**, and **Kaggle Free GPUs**
> 🔒 100% private — no external API calls required
> 🧠 Optional Cloud LLM support (OpenAI, Gemini, Groq, DeepSeek)

---

# 🧠 **Features**

### 🔍 Automatic Skill Extraction

* Extracts technical, domain, and soft skills from both **resume** and **JD**.

### 📊 Skill Gap Analysis

* Identifies matching, partial, and missing skills.

### 🎯 Learning Roadmap

* Creates a custom **study plan** + **5 real-world AI/ML project ideas**.

### 🧮 Job-Fit Score

* Predicts how well the resume matches the JD → score out of 100.

### 💻 Multi-Mode Support

* Local CPU mode
* Local GPU mode (future RTX GPUs)
* Kaggle T4 free GPU mode
* Cloud API mode (GPT-4 / Gemini / Groq / DeepSeek)

### 🛡️ Privacy

* Everything runs locally → safe for resumes and sensitive data.

---

# 🏗️ **Tech Stack**

| Component             | Technology                     |
| --------------------- | ------------------------------ |
| Embeddings            | MiniLM (SentenceTransformers)  |
| Vector DB             | ChromaDB                       |
| Framework             | LangChain v1.x (manual RAG)    |
| UI                    | Streamlit                      |
| Local LLMs            | TinyLlama / GPT-2 / Qwen       |
| Cloud LLMs (Optional) | OpenAI, Gemini, Groq, DeepSeek |
| Hardware              | CPU / GPU Auto-detect          |
| Cloud                 | Kaggle Free GPU                |

---

# 📁 **Project Structure**

```
job-analyzer-basic/
│
├── app.py                 # CLI version (CPU/GPU auto)
├── streamlit_app.py       # Streamlit UI version
│
├── sample_data/
│   ├── sample_resume.txt
│   └── sample_jd.txt
│
├── chroma_store/          # Auto-generated vector DB
├── tmp/                   # Uploaded resume/JD files
│
├── requirements.txt
└── README.md
```

---

# ⚙️ **Installation**

## 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/job-analyzer-basic.git
cd job-analyzer-basic
```

## 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏃 **How to Run**

## ▶️ Run CLI Version

```bash
python app.py
```

---

## 🌐 Run Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

Then open:

👉 [http://localhost:8501/](http://localhost:8501/)

Upload **resume.txt** and **jd.txt** → click **Run Analysis**.

---

# ⚡ **Execution Modes (CPU, GPU, Kaggle, API)**

Your code auto-detects GPU:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 🟩 1. Local CPU Mode (Default)

✔ Free
✔ Offline
✔ Works everywhere
✔ Safe for confidential resumes

❌ Slower
❌ Small models only (TinyLlama, GPT-2)

**Recommended models:**

* TinyLlama 1.1B
* GPT-2 / DistilGPT-2
* MiniLM embeddings

---

## 🟦 2. Local GPU Mode (Future GPUs — RTX cards)

(*Your GTX 1060 is too old; but this is ready for future upgrades.*)

✔ Fast inference
✔ Can run 3B–14B models
✔ Best accuracy

❌ Requires modern NVIDIA GPU

**Recommended:**

* Qwen 1.5B–4B
* Gemma 2B
* Llama 3B / 8B

---

## 🟪 3. Kaggle Free GPU Mode

Use free **Tesla T4 GPU (16GB)**.

✔ Free
✔ Runs 2B–8B models
✔ Zero setup

❌ Timeout after inactivity

**Recommended:**

* Qwen 2.5B / 4B
* Gemma 2B
* Llama 3B–8B

---

## 🔑 4. Cloud API Mode (OpenAI, Gemini, Groq, DeepSeek)

### ✔ Pros:

* Best accuracy
* Fastest processing
* No hardware needed
* Handles long resumes & large JDs

### ❌ Cons:

* Paid
* Internet required
* Privacy concerns

**Recommended Models:**

* GPT-4.1
* GPT-4o-mini
* Gemini 1.5 Pro
* Groq Llama-3-8B
* DeepSeek Chat

---

# ⚔️ **API Key vs No API Key — Side-by-Side Comparison**

| Feature        | Local (No API Key) | Cloud (API Key)    |
| -------------- | ------------------ | ------------------ |
| Cost           | Free               | Paid ($)           |
| Speed          | Medium             | Very fast          |
| Accuracy       | Medium             | Highest            |
| Privacy        | 100% Local         | Data sent to cloud |
| Hardware Needs | CPU/GPU            | None               |
| Resume Safety  | Excellent          | Medium             |
| Model Size     | ≤1.5B              | ≤100B+             |
| Setup          | Medium             | Easy               |

---

# 🏗️ **Architecture**

## ASCII Architecture Diagram

```
                      ┌─────────────────────────┐
                      │     Input Layer         │
                      │  Resume.txt + JD.txt    │
                      └────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │   Document Loaders       │
                     │  (LangChain Community)   │
                     └────────────┬─────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │      Embeddings          │
                     │  MiniLM-L6 (CPU/GPU)     │
                     └────────────┬─────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │      ChromaDB Vector      │
                     │          Store            │
                     └────────────┬─────────────┘
                                   │
                         (Top-k relevant chunks)
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │        RAG Block         │
                     │ Prompt + Retrieved Docs  │
                     └────────────┬─────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │     Local LLM Engine      │
                     │ TinyLlama / GPT2 / Qwen   │
                     │ (CPU/GPU Auto-Detect)     │
                     └────────────┬─────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │      Output Layer         │
                     │ Skills • Gaps • Roadmap   │
                     │ Job-Fit Score • Insights  │
                     └──────────────────────────┘
```

---

## Mermaid Diagram (GitHub Supported)

```mermaid
flowchart TD
    A[Resume.txt + JD.txt] --> B[Document Loaders<br>LangChain Community]
    B --> C[Embeddings<br>MiniLM-L6 (CPU/GPU)]
    C --> D[ChromaDB<br>Vector Store]
    D --> E[RAG Pipeline<br>Prompt + Retrieved Docs]
    E --> F[Local LLM<br>TinyLlama / GPT-2 / Qwen]
    F --> G[Results<br>Skills • Gaps • Roadmap • Score]
```

---

# 🧪 **Sample Output**

```
📌 JD Skills:
- Python, SQL, NLP, Transformers, AWS

📌 Resume Skills:
- Python, NLP, TensorFlow, Docker

📊 Skill Gap:
Missing → AWS, CI/CD, Airflow  
Partial → ML Ops  

🎯 Learning Roadmap:
1. AWS basics → ECS/Lambda project  
2. Build CI/CD pipeline  
3. Airflow ETL pipeline  

🧮 Job Fit Score: 78/100
```

---

# 🔧 **Troubleshooting**

### CUDA error?

Your GTX 1060 is too old → CPU fallback is automatic.

### Slow generation?

Use Kaggle free GPU (T4).

### Want more accuracy?

Use API Key mode.

---

# 🛠️ **Future Enhancements**

* PDF upload support
* Report export (PDF/HTML)
* Resume rewriting
* Model selection UI
* Dashboard with charts
* Docker container

---

# ❤️ **Contributing**

PRs are welcome.
Improve prompts, models, or add more career analytics.

---

# 📜 **License**

MIT License © 2025

---


#   S k i l l - G a p - A n a l y z e r - L o c a l - L L M - R A G - 
 
 
