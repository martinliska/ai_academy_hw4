import os
import json
import time
import re
from typing import List, Dict, Any
import pymupdf  # PyMuPDF for PDF text extraction
import whisper  # OpenAI Whisper for speech-to-text
import chromadb
from chromadb.utils import embedding_functions
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

ASSETS_DIR = "assets"
PROCESSED_DIR = "processed_data"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "genai_rag_kb"

PDF_FILE = os.path.join(ASSETS_DIR, "Databases for GenAI.pdf")
VIDEO_FILE = os.path.join(ASSETS_DIR, "2 part Databases for GenAI.mp4")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def clean_text(text: str) -> str:
    """Normalize ligatures, smart quotes, symbols, and extra whitespace for cleaner embeddings."""
    if not text:
        return ""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    # Remove icon bullet symbols that add embedding noise
    text = text.replace("🆇", "").replace("✅", "")
    return text.strip()


def extract_pdf_text(pdf_path: str = PDF_FILE) -> List[Dict[str, Any]]:
    """Step 1a: Extract text page by page from the PDF knowledge base."""
    print(f"📄 Extracting text from PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    pdf_pages = []
    doc = pymupdf.open(pdf_path)
    print(f"   Found {len(doc)} pages in PDF.")

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        raw_text = page.get_text("text").strip()
        cleaned = clean_text(raw_text)
        if cleaned:
            pdf_pages.append({
                "source": "Databases for GenAI.pdf",
                "file_type": "pdf",
                "page": page_idx + 1,
                "text": cleaned
            })

    output_json = os.path.join(PROCESSED_DIR, "pdf_text.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pdf_pages, f, indent=2, ensure_ascii=False)

    print(f"✅ PDF text extracted: {len(pdf_pages)} pages saved to {output_json}")
    return pdf_pages


def transcribe_audio_video(video_path: str = VIDEO_FILE, whisper_model_name: str = "base") -> List[Dict[str, Any]]:
    """Step 1b: Use OpenAI Whisper speech-to-text to transcribe audio/video into text."""
    cache_json = os.path.join(PROCESSED_DIR, "audio_transcription.json")

    if os.path.exists(cache_json):
        print(f"🎙️ Loading cached Whisper audio transcription from {cache_json}")
        with open(cache_json, "r", encoding="utf-8") as f:
            segments = json.load(f)
        return segments

    print(f"🎙️ Transcribing audio/video with OpenAI Whisper ('{whisper_model_name}' model): {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video/Audio file not found at {video_path}")

    start_time = time.time()
    model = whisper.load_model(whisper_model_name)
    result = model.transcribe(video_path, fp16=False)
    elapsed = time.time() - start_time
    print(f"✅ Speech-to-text transcription completed in {elapsed:.2f} seconds.")

    segments = []
    for seg in result.get("segments", []):
        start_str = format_timestamp(seg["start"])
        end_str = format_timestamp(seg["end"])
        cleaned_seg = clean_text(seg["text"])
        if cleaned_seg:
            segments.append({
                "source": "2 part Databases for GenAI.mp4",
                "file_type": "audio",
                "start_time": seg["start"],
                "end_time": seg["end"],
                "location": f"{start_str} - {end_str}",
                "text": cleaned_seg
            })

    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"✅ Audio transcription saved: {len(segments)} segments to {cache_json}")
    return segments


def chunk_text(documents: List[Dict[str, Any]], chunk_size: int = 600, overlap: int = 100) -> List[Dict[str, Any]]:
    """Step 2: Split text into smaller, semantically meaningful chunks using LlamaIndex SentenceSplitter."""
    print(f"✂️ Chunking text with LlamaIndex SentenceSplitter (target size: {chunk_size} chars, overlap: {overlap} chars)...")
    
    # Convert input dictionary items into LlamaIndex Document instances
    llama_docs = []
    for doc in documents:
        text = doc.get("text", "").strip()
        if not text:
            continue
        
        source = doc.get("source", "Unknown")
        file_type = doc.get("file_type", "unknown")
        location = f"Page {doc['page']}" if file_type == "pdf" else doc.get("location", "Audio")
        
        llama_docs.append(Document(
            text=text,
            metadata={
                "source": source,
                "file_type": file_type,
                "location": location,
                "page": doc.get("page", 0)
            }
        ))

    # Initialize LlamaIndex SentenceSplitter node parser
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents(llama_docs)

    chunks = []
    for idx, node in enumerate(nodes, 1):
        meta = node.metadata
        file_type = meta.get("file_type", "unknown")
        source = meta.get("source", "Unknown")
        location = meta.get("location", "N/A")

        chunks.append({
            "chunk_id": f"{file_type}_chunk_{idx}",
            "source": source,
            "file_type": file_type,
            "location": location,
            "text": node.get_content().strip()
        })

    output_chunks = os.path.join(PROCESSED_DIR, "text_chunks.json")
    with open(output_chunks, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Text chunking via LlamaIndex complete: {len(chunks)} chunks created and saved to {output_chunks}")
    return chunks


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build_vector_database(chunks: List[Dict[str, Any]], db_path: str = CHROMA_DB_DIR) -> chromadb.Collection:
    """Step 3: Embed chunks using BAAI/bge-small-en-v1.5 and store in ChromaDB persistent collection."""
    print(f"🧠 Embedding chunks using '{EMBEDDING_MODEL_NAME}' and storing in ChromaDB at '{db_path}'...")
    client = chromadb.PersistentClient(path=db_path)

    # Use high-performing retrieval embedding model
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # Re-create collection for clean indexing
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "GenAI Databases RAG Knowledge Base"}
    )

    documents = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "file_type": c["file_type"],
            "location": c["location"],
            "chunk_id": c["chunk_id"]
        }
        for c in chunks
    ]

    # Batch add
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=documents[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )

    print(f"✅ Successfully indexed {collection.count()} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
    return collection


def run_full_ingestion():
    """Run full end-to-end data processing and indexing pipeline."""
    print("🚀 Starting RAG Knowledge Base Ingestion Pipeline...")
    t0 = time.time()

    # 1. Load PDF & Speech-to-Text Audio
    pdf_pages = extract_pdf_text()
    audio_segments = transcribe_audio_video()

    # Combine document sources
    combined_docs = pdf_pages + audio_segments

    # 2. Chunk text
    chunks = chunk_text(combined_docs)

    # 3. Embed & Store in ChromaDB
    collection = build_vector_database(chunks)

    t1 = time.time()
    print(f"🎉 RAG Ingestion Complete in {t1 - t0:.2f} seconds!")
    return len(chunks), collection.count()


if __name__ == "__main__":
    run_full_ingestion()
