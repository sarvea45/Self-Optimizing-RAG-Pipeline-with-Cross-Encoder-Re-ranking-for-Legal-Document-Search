"""
core/retriever.py
Stage 1: ChromaDB vector similarity retrieval.
Compatible with Python 3.9+
"""

import os
import chromadb
from typing import Optional, List
from core.embedder import embed_query

COLLECTION_NAME = "legal_chunks"

_client = None   # type: Optional[object]
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        print("[Retriever] Connecting to ChromaDB at '{}'...".format(persist_dir))
        _client = chromadb.PersistentClient(path=persist_dir)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[Retriever] Collection '{}' ready ({} chunks)".format(
            COLLECTION_NAME, _collection.count()
        ))
    return _collection


def retrieve(query, top_k):
    # type: (str, int) -> List[dict]
    collection = get_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        raise RuntimeError(
            "Vector index is empty. Please run scripts/embed.py first."
        )

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

    for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
        score = float(1 - dist / 2)
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id":   meta.get("doc_id", ""),
            "text":     text,
            "score":    round(score, 6),
        })

    return chunks