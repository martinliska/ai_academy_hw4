import os
import sys
import time
import requests
import json
from typing import List, Dict, Any, Generator, Union
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "genai_rag_kb"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:latest"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def get_chroma_collection(db_path: str = CHROMA_DB_DIR) -> chromadb.Collection:
    """Connect to ChromaDB persistent collection with BAAI/bge-small-en-v1.5 embedding model."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ChromaDB directory not found at {db_path}. Please run ingest.py first.")

    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    return collection


def retrieve_context(query: str, top_k: int = 10, db_path: str = CHROMA_DB_DIR) -> List[Dict[str, Any]]:
    """Retrieve top-K most relevant chunks from ChromaDB for a user query."""
    collection = get_chroma_collection(db_path)

    # Format query with BGE instruction prefix if BGE model is used
    query_text = f"Represent this sentence for searching relevant passages: {query}" if "bge" in EMBEDDING_MODEL_NAME.lower() else query

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []
    if results and results.get("documents") and len(results["documents"]) > 0:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for idx in range(len(docs)):
            # Convert distance to approximate similarity score (Cosine distance: distance = 1 - similarity)
            dist = distances[idx]
            sim_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            retrieved_chunks.append({
                "rank": idx + 1,
                "text": docs[idx],
                "source": metas[idx].get("source", "Unknown"),
                "file_type": metas[idx].get("file_type", "unknown"),
                "location": metas[idx].get("location", "N/A"),
                "chunk_id": metas[idx].get("chunk_id", ""),
                "distance": dist,
                "similarity": sim_score
            })

    return retrieved_chunks


def build_rag_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Construct a clean, structured RAG prompt with ground context and source attributions."""
    context_str = ""
    for idx, chunk in enumerate(context_chunks, 1):
        context_str += f"\n--- [Context #{idx} | Source: {chunk['source']} ({chunk['location']})] ---\n"
        context_str += f"{chunk['text']}\n"

    prompt = f"""You are a helpful, precise AI specialist on Databases for Generative AI.
Answer the user's question accurately using ONLY the context information provided below.
If the answer cannot be determined from the context, clearly state what information is missing.
Always cite your source materials (PDF page number or Video timestamp) when presenting facts.

KNOWLEDGE BASE CONTEXT:
{context_str}

USER QUESTION: {query}

DETAILED ANSWER:"""

    return prompt


def generate_answer(
    query: str,
    top_k: int = 10,
    model: str = DEFAULT_MODEL,
    stream: bool = False
) -> Dict[str, Any]:
    """Execute the full RAG pipeline: Query -> Retrieval -> Local LLM Generation."""
    t0 = time.time()

    # Step 1: Vector Search Retrieval
    context_chunks = retrieve_context(query, top_k=top_k)
    t_retrieve = time.time()

    # Step 2: Prompt Construction
    prompt = build_rag_prompt(query, context_chunks)

    # Step 3: Local LLM Generation via Ollama
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 1024
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        answer = f"Error connecting to Ollama LLM at {OLLAMA_API_URL}: {str(e)}"

    t_gen = time.time()

    return {
        "query": query,
        "answer": answer,
        "context_chunks": context_chunks,
        "prompt": prompt,
        "model": model,
        "retrieval_time_sec": round(t_retrieve - t0, 3),
        "generation_time_sec": round(t_gen - t_retrieve, 3),
        "total_time_sec": round(t_gen - t0, 3)
    }


def get_available_ollama_models() -> List[str]:
    """Fetch installed models from local Ollama instance."""
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        if res.status_code == 200:
            models = res.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        pass
    return [DEFAULT_MODEL]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        user_input = input("Enter your RAG question (or press Enter for default): ").strip()
        test_query = user_input if user_input else "What are the main types of databases used for Generative AI?"

    print(f"\n🔍 Querying RAG Pipeline: '{test_query}'\n")
    result = generate_answer(test_query, top_k=10)
    print("🤖 ANSWER:\n", result["answer"])
    print("\n📚 RETRIEVED SOURCES:")
    for c in result["context_chunks"]:
        print(f" - [{c['source']} | {c['location']}] (Similarity: {c['similarity']:.2f})")
    print(f"\n⏱️ Total latency: {result['total_time_sec']}s (Retrieval: {result['retrieval_time_sec']}s, Generation: {result['generation_time_sec']}s)")
