"""
core/embedder.py
────────────────
Bi-encoder embedding module using sentence-transformers.

Model Choice: all-MiniLM-L6-v2
  - 22MB model, 384-dimensional embeddings
  - Top-ranked on MTEB benchmark for its size class
  - Excellent speed/accuracy trade-off for semantic search
  - Well-tested on domain-specific retrieval tasks

GPU: Automatically uses RTX 3050 if available via PyTorch CUDA detection.
"""

import os
import torch
from sentence_transformers import SentenceTransformer

# ── Singleton model instance (loaded once, reused across requests) ─────────────
_biencoder_model: SentenceTransformer | None = None


def _get_device() -> str:
    """Determine compute device based on availability and config."""
    force_cpu = os.getenv("FORCE_CPU", "0") == "1"
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_embedder() -> SentenceTransformer:
    """Return the singleton bi-encoder model, loading it if necessary."""
    global _biencoder_model
    if _biencoder_model is None:
        model_name = os.getenv("BIENCODER_MODEL", "all-MiniLM-L6-v2")
        device = _get_device()
        print(f"[Embedder] Loading '{model_name}' on {device.upper()}...")
        _biencoder_model = SentenceTransformer(model_name, device=device)
        print(f"[Embedder] Model loaded ✅")
    return _biencoder_model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Embed a list of text strings into dense vectors.

    Args:
        texts: List of strings to embed.
        batch_size: Number of texts to process per GPU/CPU batch.

    Returns:
        List of float vectors (one per input text).
    """
    model = get_embedder()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalize for cosine similarity
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string. Convenience wrapper around embed_texts."""
    return embed_texts([query])[0]
