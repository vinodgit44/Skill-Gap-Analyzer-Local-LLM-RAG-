# JD–Resume Skill Gap Analyzer (Local LLM + RAG)

------------------------------------------------------------
🚀 **Overview**
------------------------------------------------------------
A privacy-preserving **AI career analysis tool** that compares your **resume** with any **job description** using a fully local **RAG + LLM pipeline**.

Runs on:
- Local CPU
- Future Local GPU
- Kaggle T4 GPU
- Optional cloud LLMs

------------------------------------------------------------
🧠 **Features**
------------------------------------------------------------
- Automatic skill extraction  
- Skill gap analysis  
- Learning roadmap  
- Job-fit scoring  
- Local + Kaggle + Cloud modes  
- 100% local privacy  

------------------------------------------------------------
🏗️ **Tech Stack**
------------------------------------------------------------
- **Embeddings:** MiniLM  
- **Vector DB:** ChromaDB  
- **Framework:** LangChain v1.x  
- **UI:** Streamlit  
- **LLMs:** TinyLlama / GPT-2 / Qwen  
- **Cloud Models:** GPT‑4, Gemini, Groq, DeepSeek  

------------------------------------------------------------
📁 **Project Structure**
------------------------------------------------------------
```
job-analyzer-basic/
├── app.py                 # CLI version
├── streamlit_app.py       # Streamlit UI
│
├── sample_data/
│   ├── sample_resume.txt
│   └── sample_jd.txt
│
├── chroma_store/          # Vector DB
├── tmp/                   # Uploaded files
│
├── requirements.txt
└── README.md
```

------------------------------------------------------------
⚙️ **Installation**
------------------------------------------------------------

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<username>/job-analyzer-basic.git
cd job-analyzer-basic
```

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

------------------------------------------------------------
🏃 **How to Run**
------------------------------------------------------------

### ▶️ CLI Version
```bash
python app.py
```

### 🌐 Streamlit Web UI
```bash
streamlit run streamlit_app.py
```

Open browser:  
👉 http://localhost:8501/

------------------------------------------------------------
⚡ **Execution Modes**
------------------------------------------------------------

### 🟩 Local CPU (Default)
✔ Free  
✔ Offline  

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 🟦 Future Local GPU
✔ Fast  
❌ Needs modern GPU  

### 🟪 Kaggle Free GPU (T4)
✔ Free  
✔ Runs 2B–8B models  

### 🔑 Cloud API Mode
✔ Best accuracy  
✔ Fastest  

------------------------------------------------------------
🏗️ **Architecture — ASCII Diagram**
------------------------------------------------------------
```
Resume.txt + JD.txt
        │
        ▼
Document Loaders
        │
        ▼
MiniLM Embeddings
        │
        ▼
ChromaDB Vector Store
        │
        ▼
RAG Pipeline
        │
        ▼
Local LLM (TinyLlama / GPT2 / Qwen)
        │
        ▼
Skills • Gaps • Roadmap • Score
```

------------------------------------------------------------
🧪 **Sample Output**
------------------------------------------------------------
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
2. CI/CD pipeline
3. Airflow ETL workflow

🧮 Job Fit Score: 78/100
```

------------------------------------------------------------
🔧 **Troubleshooting**
------------------------------------------------------------
- GTX 1060 = CPU fallback  
- Slow? Use Kaggle T4 GPU  
- Want accuracy? Use Cloud API  

------------------------------------------------------------
🛠️ **Future Enhancements**
------------------------------------------------------------
- PDF upload  
- Report export (PDF/HTML)  
- Resume rewriting  
- Model selection UI  
- Dashboard visualizations  
- Docker image  

------------------------------------------------------------
❤️ **Contributing**
PRs welcome!

------------------------------------------------------------
📜 **License**
MIT License © 2025
