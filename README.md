# Legal Document RAG Search Pipeline

> A production-grade two-stage Retrieval-Augmented Generation (RAG) pipeline for legal document search, combining bi-encoder vector retrieval with cross-encoder re-ranking for high-precision results.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)

---

## Table of Contents

- [Overview](#overview)
- [Evaluation Results](#evaluation-results)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Setup Instructions](#setup-instructions)
- [Running the Pipeline](#running-the-pipeline)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system addresses a critical limitation of naive RAG pipelines: **similarity is not the same as relevance**. A simple vector search retrieves documents that share vocabulary with a query but may discuss them in a completely irrelevant context.

Our solution: a **two-stage retrieval pipeline**:

1. **Stage 1 — Bi-Encoder Retrieval** (Speed): Rapidly retrieves a broad candidate set using cosine similarity in ChromaDB.
2. **Stage 2 — Cross-Encoder Re-ranking** (Precision): A cross-encoder model re-scores each (query, candidate) pair jointly, providing deep semantic relevance scoring.

---

## Evaluation Results

Evaluated on 25 hand-crafted legal queries against 200 real CUAD contracts:

| Metric | Baseline (Vector Search) | Re-ranked (Two-Stage) | Improvement |
|---|---|---|---|
| **MRR@5** | 0.4440 | 0.5567 | **+25.4%** ✅ |
| **NDCG@10** | 0.7514 | 0.9812 | **+30.6%** ✅ |

The two-stage pipeline significantly outperforms simple vector search, proving that cross-encoder re-ranking adds real precision value for legal document retrieval.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  FastAPI Service (port 8001)                │
│                                             │
│  GET /api/v1/retrieve/baseline              │
│  ┌─────────────────────────────────────┐   │
│  │ Bi-Encoder (all-MiniLM-L6-v2)       │   │
│  │ → ChromaDB cosine search            │   │
│  │ → Return top-k chunks               │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  GET /api/v1/retrieve/reranked              │
│  ┌─────────────────────────────────────┐   │
│  │ Stage 1: Retrieve k×10 candidates   │   │
│  │ Stage 2: Cross-Encoder re-rank      │   │
│  │ → Return top-k by relevance score   │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
    │
    ▼
ChromaDB Vector Index
(data/chroma_db/)
```

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| API Framework | FastAPI | Async, auto-docs, Pydantic validation |
| Vector Database | ChromaDB | Persistent, metadata-native, no separate server |
| Bi-Encoder | all-MiniLM-L6-v2 | Top MTEB performance for its size class |
| Cross-Encoder | ms-marco-MiniLM-L-6-v2 | MS MARCO trained, fast, accurate re-ranking |
| ML Backend | PyTorch + CUDA 12.4 | GPU acceleration (RTX 3050 tested) |
| Dataset | CUAD (Atticus Project) | 500+ real legal contracts |
| Containerization | Docker + Compose | Reproducible, one-command startup |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/sarvea45/Self-Optimizing-RAG-Pipeline-with-Cross-Encoder-Re-ranking-for-Legal-Document-Search.git
cd Self-Optimizing-RAG-Pipeline-with-Cross-Encoder-Re-ranking-for-Legal-Document-Search

# 2. Set up environment variables
cp .env.example .env

# 3. Build and start the API
docker-compose up --build -d

# 4. Run data ingestion (inside container)
docker-compose exec api python scripts/ingest.py

# 5. Build vector index
docker-compose exec api python scripts/embed.py

# 6. Test the API
curl "http://localhost:8001/health"
curl "http://localhost:8001/api/v1/retrieve/baseline?query=termination+clause&k=5"
curl "http://localhost:8001/api/v1/retrieve/reranked?query=termination+clause&k=5"

# 7. Run evaluation (from local machine)
python scripts/evaluate.py --api-url http://localhost:8001
```

---

## Setup Instructions

### Prerequisites

- Docker Desktop (v24+) with WSL2 backend (Windows) or Docker Engine (Linux/Mac)
- NVIDIA GPU with CUDA 12.x driver (optional but recommended for speed)
- 5GB free disk space
- Python 3.11 (for running evaluate.py locally)

### Step 1: Clone and Configure

```bash
git clone https://github.com/sarvea45/Self-Optimizing-RAG-Pipeline-with-Cross-Encoder-Re-ranking-for-Legal-Document-Search.git
cd Self-Optimizing-RAG-Pipeline-with-Cross-Encoder-Re-ranking-for-Legal-Document-Search
cp .env.example .env
```

### Step 2: Build and Start

```bash
docker-compose up --build -d
```

This will:
- Build the Docker image with CUDA 12.4 support (~10-15 mins first time)
- Start the FastAPI server on port **8001**
- Run health checks every 30 seconds

Check the service is healthy:

```bash
docker-compose ps
# Should show: legal-rag-api   running (healthy)
```

### Step 3: Ingest Legal Documents

```bash
docker-compose exec api python scripts/ingest.py
```

Downloads 200 CUAD contracts from Hugging Face and produces `data/processed/chunks.jsonl`.

> **Note**: CUAD-QA dataset repeats contract text for each question. Our script correctly takes the text **once per contract** to avoid duplication.

Options:
```bash
# Process specific number of documents
docker-compose exec api python scripts/ingest.py --max-docs 50

# Custom chunk size
docker-compose exec api python scripts/ingest.py --chunk-size 512 --overlap 100
```

### Step 4: Build Vector Index

```bash
docker-compose exec api python scripts/embed.py
```

Embeds chunks and stores in ChromaDB. GPU-accelerated if available (~3 mins on RTX 3050).

Options:
```bash
# Limit chunks to embed
docker-compose exec api python scripts/embed.py --max-chunks 10000

# Reset and rebuild from scratch
docker-compose exec api python scripts/embed.py --reset
```

---

## Running the Pipeline

### Without Docker (Local Development)

```bash
# Install PyTorch with CUDA
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Copy env
cp .env.example .env

# Run ingestion
python scripts/ingest.py

# Build index
python scripts/embed.py

# Start API (port 8001)
uvicorn main:app --reload --port 8001
```

---

## API Reference

### Health Check

```
GET /health
```

Response:
```json
{"status": "ok", "service": "legal-rag-pipeline"}
```

---

### Baseline Retrieval

```
GET /api/v1/retrieve/baseline?query={query}&k={k}
```

Parameters:
- `query` (required): The legal search query
- `k` (optional, default: 10): Number of results to return

Example:
```bash
curl "http://localhost:8001/api/v1/retrieve/baseline?query=termination+clause&k=5"
```

Response:
```json
{
  "results": [
    {
      "doc_id": "cuad_0045_WPPPLC_04_30_2020-EX-4_28-SERVICE_AGREEMENT",
      "chunk_id": "cuad_0045_WPPPLC_04_30_2020-EX-4_28-SERVICE_AGREEMENT-67",
      "text": "Either party may terminate this Agreement upon thirty (30) days written notice...",
      "score": 0.847
    }
  ]
}
```

---

### Re-ranked Retrieval

```
GET /api/v1/retrieve/reranked?query={query}&k={k}
```

Same parameters as baseline. Internally retrieves `k × 10` candidates, then re-ranks with cross-encoder.

Example:
```bash
curl "http://localhost:8001/api/v1/retrieve/reranked?query=termination+clause&k=5"
```

Response schema identical to baseline, but scores reflect cross-encoder relevance (not cosine similarity).

---

### Interactive Docs

Visit `http://localhost:8001/docs` for the full Swagger UI.

---

## Evaluation

### Run Evaluation Script

```bash
# Run from local machine (NOT inside container)
python scripts/evaluate.py --api-url http://localhost:8001
```

Evaluates both pipelines against 25 hand-crafted legal queries and saves results to `results/evaluation_metrics.json`.

Options:
```bash
# Custom cutoffs
python scripts/evaluate.py --api-url http://localhost:8001 --k-mrr 5 --k-ndcg 10
```

### Actual Results

`results/evaluation_metrics.json`:
```json
{
  "baseline": {
    "mrr_at_5": 0.444,
    "ndcg_at_10": 0.7514
  },
  "reranked": {
    "mrr_at_5": 0.5567,
    "ndcg_at_10": 0.9812
  }
}
```

### Metrics Explained

- **MRR@5** (Mean Reciprocal Rank): Average of `1/rank` of the first relevant document across all queries. Score of 1.0 = always found at rank 1.
- **NDCG@10** (Normalized Discounted Cumulative Gain): Measures quality of the full top-10 ranking, penalizing relevant documents appearing lower. Score of 1.0 = perfect ranking.

---

## Project Structure

```
legal-rag-pipeline/
├── main.py                      # FastAPI app entrypoint
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container build (CUDA 12.4)
├── docker-compose.yml           # Service orchestration + GPU support
├── .env.example                 # Environment variable documentation
├── README.md                    # This file
│
├── api/
│   ├── __init__.py
│   └── routes.py                # /baseline and /reranked endpoints
│
├── core/
│   ├── __init__.py
│   ├── embedder.py              # Bi-encoder (all-MiniLM-L6-v2)
│   ├── retriever.py             # ChromaDB vector search (Stage 1)
│   └── reranker.py              # Cross-encoder re-ranking (Stage 2)
│
├── scripts/
│   ├── ingest.py                # Download CUAD → chunks.jsonl
│   ├── embed.py                 # chunks.jsonl → ChromaDB index
│   └── evaluate.py              # Compute MRR@5 & NDCG@10
│
├── data/
│   ├── raw/                     # Raw downloaded documents
│   ├── processed/
│   │   └── chunks.jsonl         # Chunked text output
│   └── chroma_db/               # Persisted ChromaDB vector index
│
├── evaluation/
│   └── queries.json             # 25 legal queries + real CUAD ground truth
│
├── results/
│   └── evaluation_metrics.json  # Final MRR@5 & NDCG@10 scores
│
└── docs/
    └── technical_analysis.md    # Chunking strategy, model selection, failure analysis
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|---|---|---|
| `BIENCODER_MODEL` | `all-MiniLM-L6-v2` | HuggingFace bi-encoder model name |
| `CROSSENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB persistence directory |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8001` | API bind port |
| `DEFAULT_K` | `10` | Default number of results |
| `CANDIDATE_MULTIPLIER` | `10` | Re-ranking candidate pool multiplier |
| `CHUNKS_PATH` | `./data/processed/chunks.jsonl` | Path to chunked data |
| `FORCE_CPU` | `0` | Set to `1` to disable GPU |

---

## Troubleshooting

**API not healthy after `docker-compose up`:**
```bash
docker-compose logs api
```
Models take ~60 seconds to load on first start.

**"Vector index is empty" error:**
```bash
docker-compose exec api python scripts/ingest.py
docker-compose exec api python scripts/embed.py
```

**Evaluation script shows no output on Windows PowerShell:**
```cmd
# Use Command Prompt (CMD) instead of PowerShell
python scripts/evaluate.py --api-url http://localhost:8001
```

**CUAD download fails:**
The ingest script automatically falls back to synthetic legal documents for demonstration.

**GPU not detected in container:**
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Port conflict on 8001:**
```bash
# Find and kill the process using the port
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```