"""
core/reranker.py
Stage 2: Cross-encoder re-ranking.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2
Compatible with Python 3.9+
"""

import os
import torch
from typing import Optional, List
from sentence_transformers import CrossEncoder

_crossencoder_model = None  # type: Optional[CrossEncoder]


def get_reranker():
    # type: () -> CrossEncoder
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
        print("[Reranker] Loading '{}' on {}...".format(model_name, device.upper()))
        _crossencoder_model = CrossEncoder(model_name, device=device)
        print("[Reranker] Model loaded")
    return _crossencoder_model


def rerank(query, candidates, top_k):
    # type: (str, List[dict], int) -> List[dict]
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, candidate["text"]) for candidate in candidates]
    raw_scores = model.predict(pairs, show_progress_bar=False)

    scored_candidates = []
    for candidate, raw_score in zip(candidates, raw_scores):
        scored = dict(candidate)
        scored["score"] = round(float(raw_score), 6)
        scored_candidates.append(scored)

    reranked = sorted(
        scored_candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    return reranked[:top_k]