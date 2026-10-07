# 🤖 Multimodal GenAI Databases RAG Chatbot

A complete, functional Retrieval-Augmented Generation (RAG) chatbot pipeline built from scratch in Python. It ingests a multimodal knowledge base (PDF documents and audio/video speech recordings), extracts and chunks text, indexes embeddings in ChromaDB, and performs RAG generation using a local Ollama LLM.

---

## 📋 Prerequisites

Before running the application, make sure you have the following installed on your machine:

1. **Python 3.10+** (Python 3.13 recommended)
2. **FFmpeg** (Required by OpenAI Whisper for audio extraction)
   ```bash
   brew install ffmpeg
   ```
3. **Ollama** (Local LLM runner)
   - Install Ollama from [ollama.com](https://ollama.com)
   - Pull a model (e.g. Gemma 4 or Llama 3):
     ```bash
     ollama pull gemma4:latest
     ```
   - Ensure Ollama service is running locally (`http://localhost:11434`).

---

## ⚙️ Installation & Setup

### 1. Clone / Open Workspace Directory
Navigate to the project root directory:
```bash
cd /path/to/hw4
```

### 2. Create and Activate Virtual Environment
```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the App (Step-by-Step)

### Step 1: Knowledge Base Data Ingestion & Processing
Ensure source files are placed in `assets/`:
- `assets/Databases for GenAI.pdf`
- `assets/2 part Databases for GenAI.mp4`

Run the data ingestion script to extract PDF text, run OpenAI Whisper speech-to-text on the audio file, chunk text, compute embeddings, and store them in ChromaDB:

```bash
python ingest.py
```
*(Note: Ingestion results are saved in `processed_data/` and `chroma_db/` for instant re-use).*

---

### Step 2: (Optional) Run CLI Pipeline Test
Test vector search retrieval and Ollama response generation directly from the command line:

```bash
python rag_pipeline.py
```

---

### Step 3: Launch Interactive Web Application UI
Start the Streamlit web chatbot interface:

```bash
streamlit run app.py
```

Or run using the virtualenv python directly:
```bash
./.venv/bin/streamlit run app.py
```

- Open your browser and navigate to: **`http://localhost:8501`**

---

## 🎛️ Using the Web Interface

1. **💬 Chatbot Tab**:
   - Ask any question regarding GenAI databases, vector search, embeddings, graph RAG, or audio lecture points.
   - Expand **"View Retrieved Context Sources & Similarity Scores"** under any response to see the exact PDF page numbers or video timestamp ranges supporting the answer.
2. **📚 Knowledge Base Explorer Tab**:
   - Browse extracted PDF text page by page.
   - Inspect Whisper audio transcript segment timestamps (`MM:SS - MM:SS`).
   - Filter and search through all 267 indexed text vector chunks.
3. **⚙️ RAG Pipeline Diagnostics Tab**:
   - Test vector search queries independently and inspect raw ChromaDB JSON metadata and similarity match scores.

---

## 🛑 How to Stop the App

To stop the Streamlit web server:
- In the active terminal window, press **`Ctrl + C`**
- Or run in terminal:
  ```bash
  pkill -f "streamlit run app.py"
  ```
