# app.py
# Single-file local "Chat with your PDFs" using embeddings + FAISS + Streamlit + optional 3D plotly view

import os
import tempfile
import pickle
import numpy as np
import streamlit as st

# Optional visual
import plotly.express as px

# PDF reader
import fitz  # PyMuPDF

# embeddings + index
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from sklearn.decomposition import PCA

# ---------- Helper utilities ----------

@st.cache_data(show_spinner=False)
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def load_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(model_name)

def extract_text_from_pdf_bytes(pdf_bytes):
    # takes bytes (uploaded file) and returns large text
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(parts)

def chunk_text(text, chunk_size=300, overlap=50):
    # Simple word-based chunker with overlap
    words = text.split()
    if len(words) <= chunk_size:
        return [ " ".join(words) ]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_faiss_index(embeddings_np):
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    return index

def safe_encode(model, texts, batch_size=32):
    # model.encode -> numpy float32
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return np.array(embs).astype("float32")

def retrieve_top_k(index, query_emb, k=3):
    D, I = index.search(np.array([query_emb]).astype("float32"), k)
    return I[0], D[0]

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Chat with your PDFs (Local)", layout="wide")
st.title("🧠 Chat with Your PDFs — Local & Private")
st.markdown("Upload a PDF and ask questions about its content. All processing happens directly in your browser, ensuring your data remains private.")

# Left: Upload area + controls
with st.sidebar:
    st.header("1. Upload Your PDF")
    st.markdown("Upload a PDF document to start chatting with its content. The text will be extracted and processed locally.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.markdown("---")
    st.header("2. Configuration Settings")

    st.subheader("Chunking Parameters")
    st.markdown("Adjust how the PDF text is divided into smaller, searchable chunks.")
    chunk_size = st.number_input("Chunk size (words)", value=300, min_value=100, max_value=2000, step=50, help="The number of words in each text chunk.")
    overlap = st.number_input("Chunk overlap (words)", value=50, min_value=0, max_value=chunk_size//2, step=10, help="The number of words that overlap between consecutive chunks.")

    st.subheader("Search Parameters")
    st.markdown("Define how many top results to retrieve and which embedding model to use.")
    
    use_reranker = st.checkbox("Enable Reranking (Cross-Encoder)", value=False, help="Use a two-stage retrieval pipeline for higher accuracy.")
    
    if use_reranker:
        top_k_initial = st.number_input("Initial top results (FAISS)", value=10, min_value=1, max_value=50, step=1)
        top_k = st.number_input("Final top results (Reranked)", value=3, min_value=1, max_value=10, step=1)
        reranker_name = st.selectbox("Reranker model", options=["cross-encoder/ms-marco-MiniLM-L-6-v2"], index=0)
    else:
        top_k = st.number_input("Number of top results (k)", value=3, min_value=1, max_value=10, step=1, help="The number of most relevant chunks to retrieve for your query.")
        top_k_initial = top_k
        
    model_name = st.selectbox("Embedding model", options=["all-MiniLM-L6-v2"], index=0, help="The model used to convert text into numerical embeddings.")
    
    st.markdown("---")
    st.subheader("Actions")
    refresh_index_btn = st.button("Re-process PDF (if settings changed)", help="Click to re-chunk and re-index the uploaded PDF with new settings.")

# Main layout: two columns
col1, col2 = st.columns([2, 1])

# Model Loading
with st.spinner("Loading embedding model..."):
    model = load_model(model_name)

if use_reranker:
    with st.spinner("Loading reranker model..."):
        reranker = load_reranker(reranker_name)

# Storage: we keep chunks, embeddings and FAISS index in session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings_np" not in st.session_state:
    st.session_state.embeddings_np = None
if "index" not in st.session_state:
    st.session_state.index = None
if "pca_points" not in st.session_state:
    st.session_state.pca_points = None

# Handle PDF upload
if uploaded_file is not None:
    # read bytes
    pdf_bytes = uploaded_file.read()
    text = extract_text_from_pdf_bytes(pdf_bytes)
    st.sidebar.success(f"Extracted {len(text.split())} words from PDF")

    # chunk
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
       
    st.sidebar.info(f"Created {len(chunks)} chunks (chunk={chunk_size}, overlap={overlap})")

    # encode
    with st.spinner("Computing embeddings..."):
        embeddings_np = safe_encode(model, chunks)

    # build index
    index = build_faiss_index(embeddings_np)

    # store
    st.session_state.chunks = chunks
    st.session_state.embeddings_np = embeddings_np
    st.session_state.index = index

    # PCA for quick 3D visualization (optional)
    try:
        pca = PCA(n_components=3)
        p3 = pca.fit_transform(embeddings_np)
        st.session_state.pca_points = p3
    except Exception:
        st.session_state.pca_points = None

    st.success("PDF indexed and ready. Ask questions in the box below.")

# If refresh button pressed and data exists, recompute PCA
if refresh_index_btn and st.session_state.embeddings_np is not None:
    try:
        pca = PCA(n_components=3)
        st.session_state.pca_points = pca.fit_transform(st.session_state.embeddings_np)
        st.success("Recomputed PCA points.")
    except Exception:
        st.warning("PCA recompute failed.")

# Query UI
query = col1.text_input("Ask a question about the uploaded PDF:", placeholder="e.g., Where is the store located?")
ask_btn = col1.button("Search")

# Show index summary
with col2:
    st.subheader("Current Index Status")
    st.info(f"**Chunks in Index:** {len(st.session_state.chunks)}")
    st.info(f"**FAISS Index Ready:** {'Yes' if st.session_state.index is not None else 'No'}")
    if st.session_state.pca_points is not None:
        st.button("Visualize Embeddings in 3D (below)", help="Click to show a 3D scatter plot of your document's embeddings.")  # for UX

# Perform search
if ask_btn and query.strip() != "":
    if st.session_state.index is None:
        st.error("No indexed PDF. Upload one first.")
    else:
        q_emb = safe_encode(model, [query])[0]
        ids, distances = retrieve_top_k(st.session_state.index, q_emb, k=top_k_initial)
        
        st.subheader("Top matches (raw chunks):")
        
        if use_reranker:
            # Reranking Step
            # 1. Fetch text chunks for retrieved ids
            retrieved_chunks = [st.session_state.chunks[i] for i in ids]
            
            # 2. Form (query, chunk) pairs
            pairs = [[query, chunk] for chunk in retrieved_chunks]
            
            # 3. Predict scores
            with st.spinner("Reranking results..."):
                cross_scores = reranker.predict(pairs)
            
            # 4. Sort by score in descending order
            sorted_indices = np.argsort(cross_scores)[::-1]
            
            # 5. Keep top K final
            final_ids = [ids[idx] for idx in sorted_indices[:top_k]]
            final_scores = [cross_scores[idx] for idx in sorted_indices[:top_k]]
            
            for rank, (i, score) in enumerate(zip(final_ids, final_scores), start=1):
                st.markdown(f"**Match {rank}** — _Reranker score_: **{score:.3f}**")
                text_preview = st.session_state.chunks[i]
                st.write(text_preview[:800] + ("..." if len(text_preview)>800 else ""))
                
            display_ids = final_ids
            
        else:
            for rank, (i, dist) in enumerate(zip(ids, distances), start=1):
                score = 1.0 / (1.0 + float(dist))  # friendly pseudo-score, convert distance -> similarity-like
                st.markdown(f"**Match {rank}** — _score approx_: **{score:.3f}**")
                text_preview = st.session_state.chunks[i]
                st.write(text_preview[:800] + ("..." if len(text_preview)>800 else ""))
                
            display_ids = ids

        # Optional: show 3D plot with highlighted points
        if st.session_state.pca_points is not None:
            pts = st.session_state.pca_points
            df_plot = {
                "x": pts[:,0].tolist(),
                "y": pts[:,1].tolist(),
                "z": pts[:,2].tolist(),
                "text": [c[:200] for c in st.session_state.chunks],
                "color": [0]*len(pts)
            }
            # highlight matched indices
            for idx in display_ids:
                df_plot["color"][idx] = 1

            fig = px.scatter_3d(df_plot, x='x', y='y', z='z', color=df_plot["color"],
                                hover_name=df_plot["text"], title="3D embedding PCA (matched points highlighted)")
            fig.update_traces(marker=dict(size=4))
            st.plotly_chart(fig, use_container_width=True)

# small footer
st.markdown("--- ")
st.caption("**Privacy Note:** This application runs entirely locally in your browser. Your documents and queries are never sent to any external servers or APIs.")
