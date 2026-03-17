"""
core/reranker.py
────────────────
Stage 2 of the two-stage pipeline: cross-encoder re-ranking.

Model Choice: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking dataset (industry standard)
  - 6-layer MiniLM — fast inference, small footprint (~84MB)
  - Processes (query, document) pairs jointly for deep relevance scoring
  - Significantly outperforms bi-encoder similarity for precision ranking

How It Works:
  - Unlike bi-encoder (independent embeddings), cross-encoder sees
    BOTH query and document together in one forward pass
  - Performs token-level attention across query-document pairs
  - Outputs a single relevance logit score per pair
  - Much more accurate but O(n) slower (fine for small candidate sets)
"""

import os
import torch
from sentence_transformers import CrossEncoder

# ── Singleton cross-encoder model ─────────────────────────────────────────────
_crossencoder_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Return the singleton cross-encoder model, loading it if necessary."""
    global _crossencoder_model
    if _crossencoder_model is None:
        model_name = os.getenv(
            "CROSSENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        device = "cuda" if (
            torch.cuda.is_available() and
            os.getenv("FORCE_CPU", "0") != "1"
        ) else "cpu"
        print(f"[Reranker] Loading '{model_name}' on {device.upper()}...")
        _crossencoder_model = CrossEncoder(model_name, device=device)
        print(f"[Reranker] Model loaded ✅")
    return _crossencoder_model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Re-rank candidate chunks using the cross-encoder for precision scoring.

    Args:
        query:      The search query string.
        candidates: List of candidate dicts from Stage 1 retriever.
                    Each must have at least 'text', 'chunk_id', 'doc_id'.
        top_k:      Number of top results to return after re-ranking.

    Returns:
        Top-k candidates sorted by cross-encoder relevance score (descending).
        Each dict has 'score' replaced with the cross-encoder score.
    """
    if not candidates:
        return []

    model = get_reranker()

    # Build (query, document_text) pairs for cross-encoder
    pairs = [(query, candidate["text"]) for candidate in candidates]

    # Score all pairs in one batch — GPU accelerated on RTX 3050
    raw_scores = model.predict(pairs, show_progress_bar=False)

    # Attach cross-encoder scores to candidates
    scored_candidates = []
    for candidate, raw_score in zip(candidates, raw_scores):
        scored = dict(candidate)          # copy to avoid mutating original
        scored["score"] = round(float(raw_score), 6)
        scored_candidates.append(scored)

    # Sort by cross-encoder score descending (highest relevance first)
    reranked = sorted(
        scored_candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    return reranked[:top_k]
