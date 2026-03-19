"""
scripts/embed.py
Reads data/processed/chunks.jsonl, generates embeddings using the bi-encoder,
and stores them in ChromaDB for fast vector retrieval.

Usage:
  python scripts/embed.py
  python scripts/embed.py --max-chunks 10000
  python scripts/embed.py --batch-size 128
  python scripts/embed.py --reset
"""

import argparse
import json
import os
import sys
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from tqdm import tqdm
import chromadb

load_dotenv()

CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", "data/processed/chunks.jsonl"))
CHROMA_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION  = "legal_chunks"
BATCH_SIZE  = 64
MAX_CHUNKS  = 10000   # default cap — enough for good evaluation


def load_chunks(path, max_chunks):
    if not path.exists():
        print("[Embed] Chunks file not found: {}".format(path))
        print("[Embed] Run scripts/ingest.py first.")
        sys.exit(1)

    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if len(chunks) >= max_chunks:
                break
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                assert "doc_id"   in chunk
                assert "chunk_id" in chunk
                assert "text"     in chunk
                chunks.append(chunk)
            except Exception as e:
                print("[Embed] Skipping line {}: {}".format(line_num, e))

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS,
                        help="Max chunks to embed (default: {})".format(MAX_CHUNKS))
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing collection and rebuild")
    args = parser.parse_args()

    print("=" * 60)
    print("  Legal RAG Pipeline — Building Vector Index")
    print("=" * 60)
    print("  Max chunks : {:,}".format(args.max_chunks))
    print("  Batch size : {}".format(args.batch_size))
    print("=" * 60)

    # Load chunks
    print("[Embed] Loading chunks from {}...".format(CHUNKS_PATH))
    chunks = load_chunks(CHUNKS_PATH, args.max_chunks)
    print("[Embed] Loaded {:,} chunks".format(len(chunks)))

    # Load model
    from core.embedder import get_embedder
    model = get_embedder()

    # Init ChromaDB
    print("[Embed] Connecting to ChromaDB at '{}'...".format(CHROMA_DIR))
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if args.reset:
        print("[Embed] Resetting collection...")
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0 and not args.reset:
        print("[Embed] Collection already has {:,} chunks.".format(existing))
        print("[Embed] Use --reset to rebuild. Exiting.")
        sys.exit(0)

    # Embed in batches
    print("[Embed] Embedding {:,} chunks on GPU...".format(len(chunks)))
    total_added = 0

    for batch_start in tqdm(
        range(0, len(chunks), args.batch_size),
        desc="Embedding"
    ):
        batch = chunks[batch_start: batch_start + args.batch_size]
        ids       = [c["chunk_id"] for c in batch]
        texts     = [c["text"]     for c in batch]
        metadatas = [{"doc_id": c["doc_id"]} for c in batch]

        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_added += len(batch)

    print("=" * 60)
    print("  Vector index built!")
    print("  Total chunks indexed : {:,}".format(total_added))
    print("  ChromaDB location    : {}".format(CHROMA_DIR))
    print("=" * 60)


if __name__ == "__main__":
    main()