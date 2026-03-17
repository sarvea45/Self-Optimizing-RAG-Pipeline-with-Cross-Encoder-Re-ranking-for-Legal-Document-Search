"""
main.py
───────
Legal RAG Pipeline — FastAPI application entrypoint.

Run locally:
  uvicorn main:app --reload --port 8000

Run with Docker:
  docker-compose up --build
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from api.routes import router


# ── Lifespan: pre-load models on startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load both models into memory when API starts."""
    print("=" * 60)
    print("  Legal RAG Pipeline — Starting Up")
    print("=" * 60)

    # Pre-load bi-encoder
    from core.embedder import get_embedder
    get_embedder()

    # Pre-load cross-encoder
    from core.reranker import get_reranker
    get_reranker()

    # Connect to ChromaDB
    try:
        from core.retriever import get_collection
        col = get_collection()
        print(f"[Startup] Vector index has {col.count()} chunks")
    except Exception as e:
        print(f"[Startup] ⚠️  ChromaDB not ready yet: {e}")
        print("[Startup] Run scripts/embed.py to build the index first.")

    print("=" * 60)
    print("  API Ready ✅  →  http://0.0.0.0:8000")
    print("  Docs       →  http://0.0.0.0:8000/docs")
    print("=" * 60)

    yield  # Application runs here

    print("[Shutdown] Cleaning up...")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Legal Document RAG Search API",
    description=(
        "Two-stage retrieval pipeline for legal document search. "
        "Combines bi-encoder vector search (Stage 1) with "
        "cross-encoder re-ranking (Stage 2) for high-precision results."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(router)


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint used by Docker healthcheck and load balancers.
    Returns 200 OK when the API is running.
    """
    return {"status": "ok", "service": "legal-rag-pipeline"}


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Legal RAG Pipeline",
        "version": "1.0.0",
        "endpoints": {
            "health":   "/health",
            "baseline": "/api/v1/retrieve/baseline?query=...&k=10",
            "reranked": "/api/v1/retrieve/reranked?query=...&k=10",
            "docs":     "/docs",
        }
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
