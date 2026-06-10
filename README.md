# AgriAssist-RAG: Rice Pest Management Assistant

## Overview

AgriAssist-RAG is a Retrieval-Augmented Generation (RAG) based NLP system designed to assist farmers in managing rice pests. The system retrieves relevant agricultural knowledge from a vector database and provides deterministic recommendations based on retrieved evidence.

## Features

* Rice pest management assistance
* FAISS-based vector search
* Sentence Transformer embeddings
* Retrieval-Augmented Generation (RAG)
* Streamlit web interface

## Technologies

* Python
* Streamlit
* FAISS
* Sentence Transformers
* NumPy

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app_full.py
```

## Project Structure

* app_full.py — Main application
* index_data/ — FAISS index and metadata
* rag_nlp.ipynb — Development notebook
* demo_results.json — Sample outputs

## Author

Yash Sarin

Department of Artificial Intelligence

Woosong University
