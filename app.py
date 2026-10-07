import os
import json
import time
import streamlit as st
import pandas as pd
from rag_pipeline import (
    generate_answer,
    retrieve_context,
    get_available_ollama_models,
    CHROMA_DB_DIR,
    COLLECTION_NAME
)
from ingest import run_full_ingestion, PROCESSED_DIR

# Page configuration
st.set_page_config(
    page_title="HW4 GenAI Databases RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Theme, Glassmorphism, Custom Badges)
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #fff !important;
    }
    
    /* Header banner */
    .header-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    /* Custom Badges */
    .source-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-pdf {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-audio {
        background: rgba(192, 132, 252, 0.15);
        color: #c084fc;
        border: 1px solid rgba(192, 132, 252, 0.3);
    }
    .badge-score {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    /* Context card container */
    .context-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .context-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .context-text {
        color: #fff;
        font-size: 0.92rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    
    /* Timing box */
    .metric-chip {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .stMarkdown {
        color: #fff;
    }
    textarea {
        color: #000 !important;
    }
    .stMainBlockContainer p {
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)


def load_processed_stats():
    """Load statistics about ingested PDF, audio, and chunks."""
    stats = {"pdf_pages": 0, "audio_segments": 0, "chunks": 0}
    try:
        pdf_path = os.path.join(PROCESSED_DIR, "pdf_text.json")
        if os.path.exists(pdf_path):
            with open(pdf_path, "r", encoding="utf-8") as f:
                stats["pdf_pages"] = len(json.load(f))

        audio_path = os.path.join(PROCESSED_DIR, "audio_transcription.json")
        if os.path.exists(audio_path):
            with open(audio_path, "r", encoding="utf-8") as f:
                stats["audio_segments"] = len(json.load(f))

        chunks_path = os.path.join(PROCESSED_DIR, "text_chunks.json")
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                stats["chunks"] = len(json.load(f))
    except Exception:
        pass
    return stats


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

stats = load_processed_stats()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/brain.png", width=64)
    st.title("RAG Controls")
    st.caption("Multi-modal Knowledge Base Chatbot")

    st.markdown("---")
    st.subheader("⚙️ Configuration")

    # LLM Model Selector
    available_models = get_available_ollama_models()
    selected_model = st.selectbox(
        "Local LLM (Ollama)",
        options=available_models,
        index=0 if available_models else None,
        help="Model running locally via Ollama server"
    )

    # Retrieval Top-K Slider
    top_k = st.slider(
        "Retrieved Chunks (Top-K)",
        min_value=1,
        max_value=15,
        value=10,
        help="Number of context chunks to retrieve from ChromaDB vector store"
    )

    st.markdown("---")
    st.subheader("📊 Knowledge Base Stats")
    col_a, col_b = st.columns(2)
    col_a.metric("PDF Pages", stats["pdf_pages"])
    col_b.metric("Audio Segments", stats["audio_segments"])
    st.metric("Indexed Chunks", stats["chunks"])

    st.markdown("---")
    st.subheader("🛠️ Maintenance")

    if st.button("🔄 Re-Ingest Knowledge Base", help="Re-run PDF extraction, Whisper transcription, and ChromaDB indexing"):
        with st.spinner("Processing PDF & Audio with Whisper... This may take a moment."):
            try:
                num_chunks, vector_count = run_full_ingestion()
                st.success(f"Ingested {num_chunks} chunks into ChromaDB ({vector_count} total vectors)!")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with PyMuPDF, OpenAI Whisper, ChromaDB, SentenceTransformers & Ollama.")

# ---------------- MAIN CONTENT ----------------
st.markdown("""
<div class="header-box">
    <div class="header-title">🤖 Multimodal RAG Chatbot</div>
    <div class="header-subtitle">
        Grounding answers on <b>Databases for GenAI.pdf</b> and <b>2 part Databases for GenAI.mp4 (Whisper Transcribed Audio)</b>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_chat, tab_kb, tab_pipeline = st.tabs(["💬 Chatbot", "📚 Knowledge Base Explorer", "⚙️ RAG Pipeline Diagnostics"])

# ---------------- TAB 1: CHATBOT ----------------
with tab_chat:
    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # If message contains retrieved context, display sources accordion
            if "context_chunks" in message and message["context_chunks"]:
                with st.expander("📚 View Retrieved Context Sources & Similarity Scores"):
                    for c in message["context_chunks"]:
                        badge_class = "badge-pdf" if c["file_type"] == "pdf" else "badge-audio"
                        st.markdown(f"""
                        <div class="context-card">
                            <div class="context-card-header">
                                <div>
                                    <span class="source-badge {badge_class}">{c['file_type'].upper()}</span>
                                    <b>{c['source']}</b> — <i>{c['location']}</i>
                                </div>
                                <span class="source-badge badge-score">Match: {c['similarity'] * 100:.1f}%</span>
                            </div>
                            <div class="context-text">{c['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    if "metrics" in message:
                        m = message["metrics"]
                        st.caption(f"⏱️ Retrieval: {m.get('retrieval', 0)}s | Generation: {m.get('generation', 0)}s | Total: {m.get('total', 0)}s")

    # Suggested Prompts
    if not st.session_state.messages:
        st.markdown("##### 💡 Suggested Questions to ask:")
        sample_queries = [
            "What are the key differences between relational databases and vector databases for GenAI?",
            "How does vector similarity search work in vector databases?",
            "What topics or points were covered in the video recording?",
            "Explain graph databases and their applications in Generative AI."
        ]
        scol1, scol2 = st.columns(2)
        for idx, q in enumerate(sample_queries):
            target_col = scol1 if idx % 2 == 0 else scol2
            if target_col.button(q, key=f"sq_{idx}"):
                st.session_state.user_input_trigger = q

    # User Input Chat Box
    user_input = st.chat_input("Ask any question about the GenAI Databases knowledge base...")

    # Triggered input from suggested buttons
    if hasattr(st.session_state, "user_input_trigger") and st.session_state.user_input_trigger:
        user_input = st.session_state.user_input_trigger
        del st.session_state.user_input_trigger

    if user_input:
        # Display user prompt
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Searching vector database & generating response with Ollama..."):
                res = generate_answer(
                    query=user_input,
                    top_k=top_k,
                    model=selected_model
                )

                st.markdown(res["answer"])

                # Display Sources Accordion
                if res["context_chunks"]:
                    with st.expander("📚 View Retrieved Context Sources & Similarity Scores"):
                        for c in res["context_chunks"]:
                            badge_class = "badge-pdf" if c["file_type"] == "pdf" else "badge-audio"
                            st.markdown(f"""
                            <div class="context-card">
                                <div class="context-card-header">
                                    <div>
                                        <span class="source-badge {badge_class}">{c['file_type'].upper()}</span>
                                        <b>{c['source']}</b> — <i>{c['location']}</i>
                                    </div>
                                    <span class="source-badge badge-score">Match: {c['similarity'] * 100:.1f}%</span>
                                </div>
                                <div class="context-text">{c['text']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.caption(f"⏱️ Retrieval: {res['retrieval_time_sec']}s | Generation: {res['generation_time_sec']}s | Total: {res['total_time_sec']}s")

        # Save assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": res["answer"],
            "context_chunks": res["context_chunks"],
            "metrics": {
                "retrieval": res["retrieval_time_sec"],
                "generation": res["generation_time_sec"],
                "total": res["total_time_sec"]
            }
        })

# ---------------- TAB 2: KNOWLEDGE BASE EXPLORER ----------------
with tab_kb:
    st.subheader("📚 Ingested Knowledge Base Content")

    kb_subtab1, kb_subtab2, kb_subtab3 = st.tabs(["📑 PDF Content", "🎙️ Audio Transcription", "🧩 Vector Chunks"])

    with kb_subtab1:
        pdf_file = os.path.join(PROCESSED_DIR, "pdf_text.json")
        if os.path.exists(pdf_file):
            with open(pdf_file, "r", encoding="utf-8") as f:
                pdf_data = json.load(f)
            st.info(f"Loaded {len(pdf_data)} pages from `Databases for GenAI.pdf`")
            for item in pdf_data:
                with st.expander(f"📄 Page {item['page']}"):
                    st.text_area("Text Content", item["text"], height=200, key=f"pdf_p_{item['page']}")
        else:
            st.warning("PDF text has not been extracted yet. Please run ingestion.")

    with kb_subtab2:
        audio_file = os.path.join(PROCESSED_DIR, "audio_transcription.json")
        if os.path.exists(audio_file):
            with open(audio_file, "r", encoding="utf-8") as f:
                audio_data = json.load(f)
            st.info(f"Loaded {len(audio_data)} transcribed audio segments from `2 part Databases for GenAI.mp4`")
            df_audio = pd.DataFrame(audio_data)
            st.dataframe(df_audio[["location", "start_time", "end_time", "text"]], use_container_width=True)
        else:
            st.warning("Audio transcription has not been generated yet. Please run ingestion.")

    with kb_subtab3:
        chunks_file = os.path.join(PROCESSED_DIR, "text_chunks.json")
        if os.path.exists(chunks_file):
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            st.info(f"Total vector chunks: {len(chunks_data)}")
            search_query = st.text_input("Filter chunks by keyword:", "")
            filtered_chunks = [c for c in chunks_data if search_query.lower() in c["text"].lower()] if search_query else chunks_data[:20]

            for chunk in filtered_chunks[:30]:
                badge_class = "badge-pdf" if chunk["file_type"] == "pdf" else "badge-audio"
                with st.expander(f"ID: {chunk['chunk_id']} | Source: {chunk['source']} ({chunk['location']})"):
                    st.write(f"**Source**: {chunk['source']} ({chunk['location']})")
                    st.write(chunk["text"])
        else:
            st.warning("No text chunks available.")

# ---------------- TAB 3: PIPELINE DIAGNOSTICS ----------------
with tab_pipeline:
    st.subheader("⚙️ End-to-End RAG Architecture & Diagnostics")
    st.markdown("""
    This application follows a complete 4-step Retrieval-Augmented Generation (RAG) architecture:
    
    1. **Data Ingestion & Processing**:
       - PDF text extracted via `PyMuPDF` (`fitz`).
       - MP4 Audio transcribed into timestamped segments using `OpenAI Whisper` (`base` speech model).
    2. **Semantic Text Chunking**:
       - Chunks text into ~500 character units with overlap, retaining metadata (`source`, `file_type`, `page`, `timestamp`).
    3. **Vector Embeddings & ChromaDB Storage**:
       - Converts chunks into dense 384-dimensional vector embeddings via `sentence-transformers/all-MiniLM-L6-v2`.
       - Persists vector indices in a local `ChromaDB` collection (`genai_rag_kb`).
    4. **Retrieval & Local LLM Generation**:
       - Top-K cosine similarity query retrieval.
       - Formats grounding prompt and passes to local `Ollama` (`gemma4:latest`).
    """)

    st.markdown("---")
    st.write("### 🧪 Vector Similarity Search Test")
    diag_query = st.text_input("Test Vector Query:", "What is vector embedding?")
    diag_topk = st.slider("Diagnostic Top-K", 1, 10, 3)

    if st.button("Run Vector Test"):
        with st.spinner("Querying ChromaDB..."):
            retrieved = retrieve_context(diag_query, top_k=diag_topk)
            st.success(f"Retrieved {len(retrieved)} chunks from ChromaDB")
            for r in retrieved:
                st.json(r)
