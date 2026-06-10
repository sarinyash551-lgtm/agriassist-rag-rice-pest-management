# app_full.py
# AgriAssist-RAG — standalone deterministic Streamlit app
# Place this file in the same folder as index_data/ (faiss.index, embeddings.npy, meta.pkl).
# Also optionally keep agri_corpus/ there (helps for debugging).

import streamlit as st
import os
import sys
import faiss
import numpy as np
import pickle
import traceback
from sentence_transformers import SentenceTransformer
import re

st.set_page_config(page_title="AgriAssist-RAG (Standalone)", layout="centered")

# --------- CONFIG: paths (change only if you put files elsewhere) ----------
INDEX_DIR = "index_data"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")
EMBED_MODEL = "all-MiniLM-L6-v2"   # lightweight embedder

# --------- HELPERS: file checks ------------------------------------------------
def missing_files():
    missing = []
    if not os.path.exists(INDEX_PATH):
        missing.append(INDEX_PATH)
    if not os.path.exists(EMB_PATH):
        missing.append(EMB_PATH)
    if not os.path.exists(META_PATH):
        missing.append(META_PATH)
    return missing

# --------- UI: top-level info --------------------------------------------------
st.title("AgriAssist-RAG — Deterministic Rice Pest Q&A")
st.write("Standalone app. Make sure `index_data/faiss.index`, `index_data/embeddings.npy`, and `index_data/meta.pkl` are in the same folder as this file.")

# --------- If required files missing, show friendly instructions -------------
missing = missing_files()
if missing:
    st.error("Required index files not found. See instructions below.")
    st.write("Missing files:")
    for p in missing:
        st.write(f"- {p}")
    st.markdown("**How to fix:**")
    st.markdown("1. In Colab, download the folder `index_data/` (contains faiss.index, embeddings.npy, meta.pkl).")
    st.markdown("2. Put that `index_data/` folder next to this `app_full.py` file.")
    st.markdown("3. Install dependencies: `pip install streamlit sentence-transformers faiss-cpu`")
    st.markdown("4. Run: `streamlit run app_full.py`")
    st.stop()

# --------- Load index and meta ------------------------------------------------
try:
    st.info("Loading FAISS index and metadata (this may take a few seconds)...")
    index = faiss.read_index(INDEX_PATH)
    embeddings = np.load(EMB_PATH)
    with open(META_PATH, "rb") as f:
        id_to_meta = pickle.load(f)
except Exception as e:
    st.error("Failed to load index or metadata. See traceback below.")
    st.text(traceback.format_exc())
    st.stop()

# --------- Load embedding model ------------------------------------------------
try:
    st.info("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer(EMBED_MODEL)
except Exception as e:
    st.error("Failed to load embedding model. Install sentence-transformers and retry.")
    st.text(str(e))
    st.stop()

# --------- Retriever function --------------------------------------------------
def retrieve(query: str, k: int = 3):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        meta = id_to_meta[int(idx)]
        results.append({"score": float(score), "text": meta["text"], "meta": meta["meta"]})
    return results

# --------- Deterministic answer builder ---------------------------------------
KEYWORDS = ["manage", "management", "control", "use", "avoid", "monitor", "apply", "release", "spray",
            "biological", "neem", "resistant", "threshold", "trichogramma", "pheromone", "sanitation"]

def extract_sentences(retrieved):
    sentences = []
    for r in retrieved:
        text = r["text"]
        parts = re.split(r'(?<=[\\.\\n])\\s+', text)
        for s in parts:
            s2 = s.strip()
            if not s2:
                continue
            if any(kw in s2.lower() for kw in KEYWORDS):
                sentences.append((s2, r["meta"]))
    return sentences

def build_answer(question, retrieved, max_actions=6):
    sents = extract_sentences(retrieved)
    if not sents:
        return "No clear guidance in the provided evidence. See retrieved chunks.", [], []
    # deduplicate preserving order
    seen = []
    dedup = []
    for s, meta in sents:
        if s not in seen:
            seen.append(s)
            dedup.append((s, meta))
    summary_sents = [s for s,_ in dedup[:3]]
    summary = " ".join(summary_sents)
    actions = []
    for s, meta in dedup:
        # split possible clauses into smaller actions
        parts = re.split(r';|, and |, then | - |,', s)
        for p in parts:
            p = p.strip()
            if len(p) > 10 and len(actions) < max_actions:
                item = p[0].upper() + p[1:]
                if item not in actions:
                    actions.append(item)
            if len(actions) >= max_actions:
                break
        if len(actions) >= max_actions:
            break
    if not actions:
        actions = summary_sents[:max_actions]
    evidence_used = []
    for i, r in enumerate(retrieved, 1):
        for s, meta in dedup:
            if meta == r["meta"] and i not in evidence_used:
                evidence_used.append(i)
    return summary.strip(), actions, evidence_used

# --------- Streamlit UI -------------------------------------------------------
question = st.text_input("Enter your rice pest question (e.g., How to manage brown planthopper?)")
k = st.sidebar.slider("Number of evidence chunks to retrieve (k)", 1, 5, 3)

if st.button("Get Answer"):
    if not question or question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving evidence..."):
            try:
                retrieved = retrieve(question, k)
            except Exception as e:
                st.error("Retrieval failed. See error below.")
                st.text(str(e))
                retrieved = []
        if not retrieved:
            st.warning("No evidence retrieved.")
        else:
            summary, actions, evidence_used = build_answer(question, retrieved)
            st.subheader("Short answer")
            st.write(summary)
            st.subheader("Recommended actions")
            for i, a in enumerate(actions, 1):
                st.write(f"{i}. {a}")
            st.subheader("Retrieved evidence (preview)")
            for i, r in enumerate(retrieved, 1):
                src = r["meta"].get("source", "unknown")
                pest = r["meta"].get("pest", "")
                st.write(f"[{i}] {src} — {pest} — score {r['score']:.3f}")
                st.write(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))
            st.caption(f"Evidence indices used: {evidence_used}")
