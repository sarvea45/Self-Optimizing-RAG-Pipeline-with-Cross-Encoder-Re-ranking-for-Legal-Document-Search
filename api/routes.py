"""
api/routes.py
─────────────
FastAPI route definitions for both retrieval endpoints.

Endpoints:
  GET /api/v1/retrieve/baseline  — Stage 1 only (vector search)
  GET /api/v1/retrieve/reranked  — Stage 1 + Stage 2 (re-ranked)

Response schema matches grading contract exactly:
  {
    "results": [
      {"doc_id": str, "chunk_id": str, "text": str, "score": float}
    ]
  }
"""

import os
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.retriever import retrieve
from core.reranker import rerank

router = APIRouter(prefix="/api/v1/retrieve")

CANDIDATE_MULTIPLIER = int(os.getenv("CANDIDATE_MULTIPLIER", "10"))
DEFAULT_K = int(os.getenv("DEFAULT_K", "10"))


# ── Response Schema (matches grading contract exactly) ────────────────────────

class ChunkResult(BaseModel):
    doc_id:   str
    chunk_id: str
    text:     str
    score:    float


class SearchResponse(BaseModel):
    results: list[ChunkResult]


# ── Baseline Endpoint ─────────────────────────────────────────────────────────

@router.get("/baseline", response_model=SearchResponse)
def baseline_search(
    query: str = Query(..., description="The legal search query text"),
    k: int = Query(DEFAULT_K, ge=1, le=100, description="Number of results to return"),
):
    """
    Single-stage vector retrieval using bi-encoder similarity search.

    - Embeds query with all-MiniLM-L6-v2
    - Searches ChromaDB using cosine similarity
    - Returns top-k chunks sorted by similarity score (descending)

    Fast but may miss subtle semantic relevance nuances.
    """
    try:
        results = retrieve(query, top_k=k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    return SearchResponse(results=[
        ChunkResult(
            doc_id=r["doc_id"],
            chunk_id=r["chunk_id"],
            text=r["text"],
            score=r["score"],
        )
        for r in results
    ])


# ── Re-ranked Endpoint ────────────────────────────────────────────────────────

@router.get("/reranked", response_model=SearchResponse)
def reranked_search(
    query: str = Query(..., description="The legal search query text"),
    k: int = Query(DEFAULT_K, ge=1, le=100, description="Number of results to return"),
):
    """
    Two-stage retrieval with cross-encoder re-ranking.

    Stage 1 — Retrieval (Broad Recall):
      - Retrieves k × CANDIDATE_MULTIPLIER candidates via vector search
      - Goal: ensure truly relevant documents are in the candidate pool

    Stage 2 — Re-ranking (High Precision):
      - Cross-encoder scores each (query, candidate) pair jointly
      - Token-level attention gives much more accurate relevance scores
      - Returns top-k by cross-encoder score (descending)

    Significantly higher precision than baseline for complex legal queries.
    """
    candidate_size = k * CANDIDATE_MULTIPLIER

    try:
        candidates = retrieve(query, top_k=candidate_size)
        results    = rerank(query, candidates, top_k=k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return SearchResponse(results=[
        ChunkResult(
            doc_id=r["doc_id"],
            chunk_id=r["chunk_id"],
            text=r["text"],
            score=r["score"],
        )
        for r in results
    ])
