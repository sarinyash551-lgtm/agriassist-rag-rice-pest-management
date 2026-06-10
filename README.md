AgriAssist-RAG: Retrieval-Augmented Q&A System for Rice Pest Management
Yash Sarin (202480201)





1. Required Libraries / Dependencies

This project uses the following Python libraries:

- streamlit
- sentence-transformers
- faiss-cpu
- pdfplumber
- transformers
- numpy
- json
- urllib
- pickle (for saving metadata)

Optional (used in the notebook):
- matplotlib


`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

2. Installation & Setup

Ensure Python 3.9+ is installed.

Install dependencies:
- pip install streamlit sentence-transformers faiss-cpu pdfplumber transformers numpy
- If `faiss-cpu` fails on Windows, use:
- pip install faiss-cpu==1.7.4


Project folder should look like:


AgriAssist/
│
├── rag_nlp.ipynb
├── app.py
├── demo_results.json
│
├── download_pdfs/
├── agri_corpus/
└── index_data/

These folders are automatically created by the notebook.

`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````	

 3. How to Run / Deploy the Code

✔ Running the Notebook (Google Colab or Jupyter)

1. Open `rag_nlp.ipynb`.
2. Run all cells **top to bottom**.
3. The notebook will:
   - Download agricultural PDFs  
   - Extract and clean text  
   - Chunk the text  
   - Generate embeddings  
   - Build a FAISS index  
   - Run a Q&A demo  
   - Save all processed files  

✔ Running the Streamlit App (Local Deployment)

In the project folder, run:

- streamlit run app.py
- Your browser will open:
- http://localhost:8501

Ask questions like:

> “How to manage brown planthopper in rice?”

End of README

