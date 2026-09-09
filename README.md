# Semantic Search Bot 🧠

A **fully local, privacy-first "Chat with your PDFs"** application built as a single-file Streamlit app. It allows you to upload a PDF document, semantically search its content using vector embeddings, and optionally visualize the document embeddings in an interactive 3D scatter plot. 

Everything runs **entirely on your machine**, ensuring your data remains private and secure without the need for external API keys or cloud servers.

## Features ✨

- **100% Local & Private**: No data leaves your machine. Uses `sentence-transformers` for local embeddings.
- **Two-Stage Retrieval (Optional)**: Employs a **Retriever-Ranker architecture**. Quickly fetch initial results with FAISS (Bi-Encoder), then rerank them with a highly accurate Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for state-of-the-art semantic search accuracy.
- **Configurable Search parameters**: Adjust text chunk size, chunk overlap, and the number of top-k results directly from the UI.
- **Interactive 3D Visualization**: Displays a 3D scatter plot of your document's embeddings using PCA dimensionality reduction, highlighting the chunks that matched your query.
- **Efficient Vector Search**: Uses FAISS (Facebook AI Similarity Search) for fast and scalable similarity matching.
- **Session Caching**: Smooth experience without unnecessary re-processing of the PDF on every query.

## Architecture 🏗️

### Standard Retrieval
```text
PDF Upload → Text Extraction → Chunking → Embedding (Bi-Encoder) → FAISS Index → Semantic Search → 3D Visualization
```

### Two-Stage Retrieval (Reranking Enabled)
```text
PDF Upload → Text Extraction → Chunking → Embedding (Bi-Encoder) → FAISS Index → Fetch Initial Top-K → Rerank with Cross-Encoder → Return Final Top-K
```

## Prerequisites

- **Python 3.10** or higher
- Git

## Installation & Setup 🛠️

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repository-url>
   cd "Semantic Search Bot"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On **Windows**:
     ```powershell
     .\venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run ▶️

Once your virtual environment is activated and dependencies are installed, you can start the application by running:

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

## How to Use 📖

1. **Upload a PDF**: Use the sidebar to upload any PDF document you want to query.
2. **Configure Settings**: Optionally tweak the *Chunking* and *Search* parameters in the sidebar to refine how the text is processed. **Enable Reranking** for more accurate results using a Cross-Encoder.
3. **Ask Questions**: In the main panel, type a question related to the uploaded PDF (e.g., *"What are the key findings?"* or *"Summarize the methodology"*).
4. **View Matches**: Click **Search** to retrieve the most relevant text chunks along with a pseudo-similarity score.
5. **Visualize**: Check out the interactive 3D plot to see how your document's text is distributed semantically, with your matching chunks highlighted in red!

## Technologies Used 💻

- **[Streamlit](https://streamlit.io/)**: Web UI framework
- **[SentenceTransformers](https://sbert.net/)**: Text embeddings (`all-MiniLM-L6-v2`)
- **[FAISS](https://github.com/facebookresearch/faiss)**: Vector similarity search index
- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/en/latest/)**: PDF text extraction
- **[Plotly](https://plotly.com/python/) & [scikit-learn](https://scikit-learn.org/)**: 3D PCA visualization
