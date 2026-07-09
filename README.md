# 🌾 AgriAssist RAG: Rice Pest Management Assistant

![Python](https://img.shields.io/badge/Python-3.x-blue)
![RAG](https://img.shields.io/badge/RAG-Enabled-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![License](https://img.shields.io/badge/License-MIT-success)

## 📌 Overview

AgriAssist is an AI-powered Retrieval-Augmented Generation (RAG) application that assists farmers by answering rice pest management queries using semantic search over an agricultural knowledge base.

The system retrieves the most relevant information using FAISS vector search before generating accurate, context-aware responses.

---

## ✨ Features

- 🌾 Rice pest diagnosis
- 🔍 Semantic search using FAISS
- 🤖 Retrieval-Augmented Generation (RAG)
- ⚡ Fast document retrieval
- 💻 Interactive Streamlit application

---

## 🛠 Tech Stack

- Python
- FAISS
- Sentence Transformers
- NumPy
- Streamlit
- Pickle

---

## 📂 Project Structure

```text
agriassist-rag-rice-pest-management/

├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
│
├── faiss_index/
│   ├── embeddings.npy
│   ├── meta.pkl
│   └── faiss.index
│
├── evaluation_results.json
│
└── agriassist_rag_pipeline.ipynb
```

---

## 🚀 Installation

```bash
git clone <repository-url>
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

- Hybrid Search (BM25 + FAISS)
- Cross Encoder Re-ranking
- LLM Integration
- Multi-language Support
- Docker Deployment

---

## 👨‍💻 Author

Yash Sarin

---

## 📄 License

MIT License
