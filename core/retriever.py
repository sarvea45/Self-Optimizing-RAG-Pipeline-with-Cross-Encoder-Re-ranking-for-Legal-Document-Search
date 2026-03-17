"""
core/retriever.py
─────────────────
Stage 1 of the two-stage pipeline: vector similarity retrieval using ChromaDB.

ChromaDB Choice:
  - Persistent client stores index to disk automatically
  - Native metadata storage (doc_id, chunk_id, text) — no separate DB needed
  - Cosine similarity via HNSW index
  - Perfect fit for 50k-chunk corpora (no need for FAISS at this scale)
"""

import os
import chromadb
from typing import Optional, List
from core.embedder import embed_query

COLLECTION_NAME = "legal_chunks"

# ── Singleton ChromaDB client + collection ────────────────────────────────────
_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_collection():
    """Return the ChromaDB collection, initializing client if necessary."""
    global _client, _collection
    if _collection is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        print(f"[Retriever] Connecting to ChromaDB at '{persist_dir}'...")
        _client = chromadb.PersistentClient(path=persist_dir)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[Retriever] Collection '{COLLECTION_NAME}' ready "
              f"({_collection.count()} chunks) ✅")
    return _collection


def retrieve(query: str, top_k: int) -> List[dict]:
    """
    Retrieve top_k chunks most similar to query using cosine vector search.

    Args:
        query: The search query string.
        top_k: Number of results to return.

    Returns:
        List of dicts sorted by similarity score (descending):
        [{"chunk_id": str, "doc_id": str, "text": str, "score": float}, ...]
    """
    collection = get_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        raise RuntimeError(
            "Vector index is empty. Please run scripts/embed.py first."
        )

    # Cap top_k to available chunks
    n_results = min(top_k, total_chunks)

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    ids       = results["ids"][0]
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (chunk_id, text, meta, dist) in enumerate(
        zip(ids, docs, metas, distances)
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score in [0, 1]
        score = float(1 - dist / 2)
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id":   meta.get("doc_id", ""),
            "text":     text,
            "score":    round(score, 6),
        })

    # Already sorted by distance (ascending) → similarity (descending)
    return chunks