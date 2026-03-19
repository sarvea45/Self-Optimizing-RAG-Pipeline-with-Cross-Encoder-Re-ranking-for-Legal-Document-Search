"""
scripts/evaluate.py
───────────────────
Evaluation script for the Legal RAG Pipeline.

Loads evaluation/queries.json, calls both API endpoints for each query,
and computes MRR@5 and NDCG@10 from scratch (no external eval libraries).

Metrics Implemented From Scratch:
  MRR@5  — Mean Reciprocal Rank at cutoff 5
  NDCG@10 — Normalized Discounted Cumulative Gain at cutoff 10

Output: results/evaluation_metrics.json

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --api-url http://localhost:8000
  python scripts/evaluate.py --k-mrr 5 --k-ndcg 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
QUERIES_PATH = Path("evaluation/queries.json")
RESULTS_PATH = Path("results/evaluation_metrics.json")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_K_MRR   = 5
DEFAULT_K_NDCG  = 10
REQUEST_TIMEOUT = 30.0   # seconds per API call


# ── Metric Implementations (from scratch, no external libraries) ───────────────

def reciprocal_rank(retrieved_doc_ids: list[str],
                    relevant_doc_ids: set[str],
                    k: int) -> float:
    """
    Compute Reciprocal Rank for a single query.

    RR = 1 / rank_of_first_relevant_document
    If no relevant document found in top-k, RR = 0.

    Args:
        retrieved_doc_ids: Ordered list of retrieved doc_ids (rank 1 first).
        relevant_doc_ids:  Set of ground-truth relevant doc_ids.
        k:                 Cutoff rank (only consider top-k results).

    Returns:
        Reciprocal rank score in [0, 1].

    Example:
        retrieved = ["doc_A", "doc_B", "doc_C"]
        relevant  = {"doc_C"}
        k = 5
        → first relevant at rank 3 → RR = 1/3 ≈ 0.333
    """
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_doc_ids: list[str],
             relevant_doc_ids: set[str],
             k: int) -> float:
    """
    Compute Discounted Cumulative Gain at cutoff k.

    DCG@k = Σ rel_i / log2(i + 1)  for i in 1..k
    where rel_i = 1 if document at rank i is relevant, else 0.

    The log2 discount penalizes relevant documents appearing lower in ranking.

    Args:
        retrieved_doc_ids: Ordered list of retrieved doc_ids (rank 1 first).
        relevant_doc_ids:  Set of ground-truth relevant doc_ids.
        k:                 Cutoff rank.

    Returns:
        DCG score (unbounded, higher is better).
    """
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            # Binary relevance: rel = 1
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved_doc_ids: list[str],
              relevant_doc_ids: set[str],
              k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at cutoff k.

    NDCG@k = DCG@k / IDCG@k
    where IDCG@k is the DCG of the ideal (perfect) ranking.

    Ideal ranking places all relevant documents at the top positions.

    Args:
        retrieved_doc_ids: Ordered list of retrieved doc_ids.
        relevant_doc_ids:  Set of ground-truth relevant doc_ids.
        k:                 Cutoff rank.

    Returns:
        NDCG score in [0, 1]. 1.0 = perfect ranking.

    Example:
        retrieved = ["doc_X", "doc_A", "doc_Y", "doc_B", "doc_Z", "doc_C"]
        relevant  = {"doc_A", "doc_B", "doc_C"}
        k = 10
        DCG  = 1/log2(3) + 1/log2(5) + 1/log2(7) = 0.631 + 0.431 + 0.356 = 1.418
        IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.000 + 0.631 + 0.500 = 2.131
        NDCG = 1.418 / 2.131 = 0.665
    """
    actual_dcg = dcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)

    # Ideal DCG: place all relevant docs at top positions
    n_relevant = min(len(relevant_doc_ids), k)
    ideal_retrieved = list(relevant_doc_ids)[:n_relevant]
    ideal_dcg = dcg_at_k(ideal_retrieved, relevant_doc_ids, k)

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def mean_reciprocal_rank(all_rr_scores: list[float]) -> float:
    """MRR = average of reciprocal rank scores over all queries."""
    if not all_rr_scores:
        return 0.0
    return sum(all_rr_scores) / len(all_rr_scores)


def mean_ndcg(all_ndcg_scores: list[float]) -> float:
    """Mean NDCG = average of NDCG scores over all queries."""
    if not all_ndcg_scores:
        return 0.0
    return sum(all_ndcg_scores) / len(all_ndcg_scores)


# ── API Client ─────────────────────────────────────────────────────────────────

def call_api(client: httpx.Client,
             endpoint: str,
             query: str,
             k: int) -> list[str]:
    """
    Call a retrieval endpoint and return ordered list of doc_ids.

    Args:
        client:   httpx client instance.
        endpoint: Full URL of the endpoint.
        query:    Search query text.
        k:        Number of results to request.

    Returns:
        Ordered list of doc_ids from the API response.
    """
    try:
        response = client.get(
            endpoint,
            params={"query": query, "k": k},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return [r["doc_id"] for r in data.get("results", [])]
    except httpx.TimeoutException:
        print(f"    ⚠️  Timeout calling {endpoint}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"    ⚠️  HTTP {e.response.status_code} from {endpoint}")
        return []
    except Exception as e:
        print(f"    ⚠️  Error calling {endpoint}: {e}")
        return []


# ── Main Evaluation Loop ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline with MRR@5 and NDCG@10"
    )
    parser.add_argument(
        "--api-url", default=DEFAULT_API_URL,
        help=f"Base URL of the API (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--k-mrr", type=int, default=DEFAULT_K_MRR,
        help=f"Cutoff k for MRR (default: {DEFAULT_K_MRR})"
    )
    parser.add_argument(
        "--k-ndcg", type=int, default=DEFAULT_K_NDCG,
        help=f"Cutoff k for NDCG (default: {DEFAULT_K_NDCG})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Legal RAG Pipeline — Evaluation")
    print("=" * 60)
    print(f"  API URL   : {args.api_url}")
    print(f"  MRR@{args.k_mrr}    : evaluating...")
    print(f"  NDCG@{args.k_ndcg}  : evaluating...")
    print("=" * 60)

    # ── Load queries ───────────────────────────────────────────────────────────
    if not QUERIES_PATH.exists():
        print(f"❌ Queries file not found: {QUERIES_PATH}")
        sys.exit(1)

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"[Eval] Loaded {len(queries)} queries from {QUERIES_PATH}")

    # ── Check API health ───────────────────────────────────────────────────────
    with httpx.Client() as client:
        try:
            health = client.get(f"{args.api_url}/health", timeout=10)
            health.raise_for_status()
            print(f"[Eval] ✅ API is healthy at {args.api_url}")
        except Exception as e:
            print(f"[Eval] ❌ API not reachable: {e}")
            print("[Eval]    Start the API with: docker-compose up  OR  uvicorn main:app")
            sys.exit(1)

    # ── Evaluation endpoints ───────────────────────────────────────────────────
    baseline_url = f"{args.api_url}/api/v1/retrieve/baseline"
    reranked_url = f"{args.api_url}/api/v1/retrieve/reranked"

    # Use max k for API calls so both metrics can be computed
    k_max = max(args.k_mrr, args.k_ndcg)

    baseline_rr_scores   = []
    baseline_ndcg_scores = []
    reranked_rr_scores   = []
    reranked_ndcg_scores = []

    per_query_results = []

    with httpx.Client() as client:
        for i, query_obj in enumerate(queries, 1):
            query_id    = query_obj["query_id"]
            query_text  = query_obj["query_text"]
            relevant    = set(query_obj["relevant_docs"])

            print(f"\n[Query {i:02d}/{len(queries)}] {query_text[:70]}...")

            # ── Baseline ───────────────────────────────────────────────────────
            t0 = time.time()
            baseline_docs = call_api(client, baseline_url, query_text, k_max)
            baseline_time = time.time() - t0

            b_rr   = reciprocal_rank(baseline_docs, relevant, args.k_mrr)
            b_ndcg = ndcg_at_k(baseline_docs, relevant, args.k_ndcg)

            baseline_rr_scores.append(b_rr)
            baseline_ndcg_scores.append(b_ndcg)

            print(f"  Baseline  → RR@{args.k_mrr}: {b_rr:.3f} | "
                  f"NDCG@{args.k_ndcg}: {b_ndcg:.3f} | {baseline_time:.2f}s")

            # ── Re-ranked ──────────────────────────────────────────────────────
            t0 = time.time()
            reranked_docs = call_api(client, reranked_url, query_text, k_max)
            reranked_time = time.time() - t0

            r_rr   = reciprocal_rank(reranked_docs, relevant, args.k_mrr)
            r_ndcg = ndcg_at_k(reranked_docs, relevant, args.k_ndcg)

            reranked_rr_scores.append(r_rr)
            reranked_ndcg_scores.append(r_ndcg)

            print(f"  Re-ranked → RR@{args.k_mrr}: {r_rr:.3f} | "
                  f"NDCG@{args.k_ndcg}: {r_ndcg:.3f} | {reranked_time:.2f}s")

            per_query_results.append({
                "query_id":        query_id,
                "query_text":      query_text,
                "baseline_rr":     round(b_rr, 6),
                "baseline_ndcg":   round(b_ndcg, 6),
                "reranked_rr":     round(r_rr, 6),
                "reranked_ndcg":   round(r_ndcg, 6),
                "baseline_docs":   baseline_docs[:args.k_ndcg],
                "reranked_docs":   reranked_docs[:args.k_ndcg],
                "relevant_docs":   list(relevant),
            })

    # ── Compute Final Metrics ──────────────────────────────────────────────────
    final_metrics = {
        "baseline": {
            "mrr_at_5":   round(mean_reciprocal_rank(baseline_rr_scores), 6),
            "ndcg_at_10": round(mean_ndcg(baseline_ndcg_scores), 6),
        },
        "reranked": {
            "mrr_at_5":   round(mean_reciprocal_rank(reranked_rr_scores), 6),
            "ndcg_at_10": round(mean_ndcg(reranked_ndcg_scores), 6),
        },
    }

    # ── Save Results ───────────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # Save detailed per-query results alongside
    detailed_path = RESULTS_PATH.parent / "evaluation_detailed.json"
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump({"summary": final_metrics, "per_query": per_query_results},
                  f, indent=2)

    # ── Print Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Baseline':>12} {'Re-ranked':>12} {'Δ Improvement':>15}")
    print("-" * 60)

    mrr_b  = final_metrics["baseline"]["mrr_at_5"]
    mrr_r  = final_metrics["reranked"]["mrr_at_5"]
    ndcg_b = final_metrics["baseline"]["ndcg_at_10"]
    ndcg_r = final_metrics["reranked"]["ndcg_at_10"]

    print(f"  {'MRR@5':<20} {mrr_b:>12.4f} {mrr_r:>12.4f} "
          f"{(mrr_r - mrr_b):>+14.4f}")
    print(f"  {'NDCG@10':<20} {ndcg_b:>12.4f} {ndcg_r:>12.4f} "
          f"{(ndcg_r - ndcg_b):>+14.4f}")
    print("=" * 60)
    print(f"  ✅ Results saved to {RESULTS_PATH}")
    print(f"  📊 Detailed results saved to {detailed_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()