"""
scripts/embed.py
────────────────
Reads data/processed/chunks.jsonl, generates embeddings using the bi-encoder,
and stores them in ChromaDB for fast vector retrieval.

Usage:
  python scripts/embed.py
  python scripts/embed.py --batch-size 128   # larger batches for more VRAM
  python scripts/embed.py --reset            # clear existing index and rebuild
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import chromadb

load_dotenv()

CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", "data/processed/chunks.jsonl"))
CHROMA_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION  = "legal_chunks"
BATCH_SIZE  = 64    # chunks per embedding batch


def load_chunks(path: Path) -> list[dict]:
    """Load all chunks from JSONL file."""
    if not path.exists():
        print(f"[Embed] ❌ Chunks file not found: {path}")
        print("[Embed]    Run scripts/ingest.py first.")
        sys.exit(1)

    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                # Validate required fields
                assert "doc_id"   in chunk, f"Missing doc_id on line {line_num}"
                assert "chunk_id" in chunk, f"Missing chunk_id on line {line_num}"
                assert "text"     in chunk, f"Missing text on line {line_num}"
                chunks.append(chunk)
            except (json.JSONDecodeError, AssertionError) as e:
                print(f"[Embed] ⚠️  Skipping malformed line {line_num}: {e}")

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Embed chunks and store in ChromaDB"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Embedding batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing collection and rebuild from scratch"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Legal RAG Pipeline — Building Vector Index")
    print("=" * 60)

    # ── Load chunks ────────────────────────────────────────────────────────────
    print(f"[Embed] Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"[Embed] ✅ Loaded {len(chunks):,} chunks")

    # ── Load bi-encoder model ──────────────────────────────────────────────────
    from core.embedder import get_embedder
    model = get_embedder()

    # ── Initialize ChromaDB ────────────────────────────────────────────────────
    print(f"[Embed] Connecting to ChromaDB at '{CHROMA_DIR}'...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if args.reset:
        print(f"[Embed] Resetting collection '{COLLECTION}'...")
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0 and not args.reset:
        print(f"[Embed] ℹ️  Collection already has {existing_count:,} chunks.")
        print("[Embed]    Use --reset to rebuild. Exiting.")
        sys.exit(0)

    # ── Embed and store in batches ─────────────────────────────────────────────
    print(f"[Embed] Embedding {len(chunks):,} chunks "
          f"(batch size: {args.batch_size})...")
    print("[Embed] GPU will be used if available (RTX 3050 detected)")

    total_added = 0

    for batch_start in tqdm(
        range(0, len(chunks), args.batch_size),
        desc="Embedding batches"
    ):
        batch = chunks[batch_start : batch_start + args.batch_size]

        ids       = [c["chunk_id"] for c in batch]
        texts     = [c["text"]     for c in batch]
        metadatas = [{"doc_id": c["doc_id"]} for c in batch]

        # Generate embeddings (GPU-accelerated via sentence-transformers)
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        # Add to ChromaDB
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_added += len(batch)

    print("=" * 60)
    print(f"  ✅ Vector index built successfully!")
    print(f"  Total chunks indexed : {total_added:,}")
    print(f"  ChromaDB location    : {CHROMA_DIR}")
    print(f"  Collection name      : {COLLECTION}")
    print("=" * 60)


if __name__ == "__main__":
    main()
