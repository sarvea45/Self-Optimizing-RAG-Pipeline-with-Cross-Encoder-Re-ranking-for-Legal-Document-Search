"""
core/embedder.py
Bi-encoder embedding module using sentence-transformers.
Model: all-MiniLM-L6-v2
Compatible with Python 3.9+
"""

import os
import torch
from typing import Optional, List
from sentence_transformers import SentenceTransformer

_biencoder_model = None  # type: Optional[SentenceTransformer]


def _get_device():
    # type: () -> str
    force_cpu = os.getenv("FORCE_CPU", "0") == "1"
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_embedder():
    # type: () -> SentenceTransformer
    global _biencoder_model
    if _biencoder_model is None:
        model_name = os.getenv("BIENCODER_MODEL", "all-MiniLM-L6-v2")
        device = _get_device()
        print("[Embedder] Loading '{}' on {}...".format(model_name, device.upper()))
        _biencoder_model = SentenceTransformer(model_name, device=device)
        print("[Embedder] Model loaded")
    return _biencoder_model


def embed_texts(texts, batch_size=64):
    # type: (List[str], int) -> List[List[float]]
    model = get_embedder()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(query):
    # type: (str) -> List[float]
    return embed_texts([query])[0]