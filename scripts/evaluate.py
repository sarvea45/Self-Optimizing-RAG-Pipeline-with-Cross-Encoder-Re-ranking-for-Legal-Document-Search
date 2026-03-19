"""
scripts/evaluate.py
Evaluation script: MRR@5 and NDCG@10 from scratch.
Calls both API endpoints and saves results/evaluation_metrics.json

Usage:
  python scripts/evaluate.py --api-url http://localhost:8001
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
QUERIES_PATH    = Path("evaluation/queries.json")
RESULTS_PATH    = Path("results/evaluation_metrics.json")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_API_URL = "http://localhost:8001"
DEFAULT_K_MRR   = 5
DEFAULT_K_NDCG  = 10
REQUEST_TIMEOUT = 60.0


# ── Metric Implementations ─────────────────────────────────────────────────────

def reciprocal_rank(retrieved_doc_ids, relevant_doc_ids, k):
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    actual_dcg = dcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)
    n_relevant  = min(len(relevant_doc_ids), k)
    ideal_dcg   = dcg_at_k(list(relevant_doc_ids)[:n_relevant], relevant_doc_ids, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


# ── API Client ─────────────────────────────────────────────────────────────────

def call_api(client, endpoint, query, k):
    try:
        response = client.get(
            endpoint,
            params={"query": query, "k": k},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return [r["doc_id"] for r in data.get("results", [])]
    except Exception as e:
        print("    Warning: API call failed: {}".format(e))
        return []


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--k-mrr",  type=int, default=DEFAULT_K_MRR)
    parser.add_argument("--k-ndcg", type=int, default=DEFAULT_K_NDCG)
    args = parser.parse_args()

    print("=" * 60)
    print("  Legal RAG Pipeline - Evaluation")
    print("=" * 60)
    print("  API URL  : {}".format(args.api_url))
    print("  MRR@{}   : evaluating...".format(args.k_mrr))
    print("  NDCG@{}  : evaluating...".format(args.k_ndcg))
    print("=" * 60)

    # Load queries
    if not QUERIES_PATH.exists():
        print("ERROR: Queries file not found: {}".format(QUERIES_PATH))
        sys.exit(1)

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print("[Eval] Loaded {} queries".format(len(queries)))

    # Check API health
    with httpx.Client() as client:
        try:
            health = client.get("{}/health".format(args.api_url), timeout=10)
            health.raise_for_status()
            print("[Eval] API is healthy at {}".format(args.api_url))
        except Exception as e:
            print("[Eval] ERROR: API not reachable: {}".format(e))
            print("[Eval] Make sure docker-compose is running and use --api-url http://localhost:8001")
            sys.exit(1)

    baseline_url = "{}/api/v1/retrieve/baseline".format(args.api_url)
    reranked_url = "{}/api/v1/retrieve/reranked".format(args.api_url)
    k_max = max(args.k_mrr, args.k_ndcg)

    baseline_rr_scores   = []
    baseline_ndcg_scores = []
    reranked_rr_scores   = []
    reranked_ndcg_scores = []
    per_query_results    = []

    with httpx.Client() as client:
        for i, query_obj in enumerate(queries, 1):
            query_id   = query_obj["query_id"]
            query_text = query_obj["query_text"]
            relevant   = set(query_obj["relevant_docs"])

            print("\n[Query {}/{}] {}...".format(i, len(queries), query_text[:60]))

            # Baseline
            t0 = time.time()
            baseline_docs = call_api(client, baseline_url, query_text, k_max)
            baseline_time = time.time() - t0
            b_rr   = reciprocal_rank(baseline_docs, relevant, args.k_mrr)
            b_ndcg = ndcg_at_k(baseline_docs, relevant, args.k_ndcg)
            baseline_rr_scores.append(b_rr)
            baseline_ndcg_scores.append(b_ndcg)
            print("  Baseline  -> RR@{}: {:.3f} | NDCG@{}: {:.3f} | {:.2f}s".format(
                args.k_mrr, b_rr, args.k_ndcg, b_ndcg, baseline_time))

            # Reranked
            t0 = time.time()
            reranked_docs = call_api(client, reranked_url, query_text, k_max)
            reranked_time = time.time() - t0
            r_rr   = reciprocal_rank(reranked_docs, relevant, args.k_mrr)
            r_ndcg = ndcg_at_k(reranked_docs, relevant, args.k_ndcg)
            reranked_rr_scores.append(r_rr)
            reranked_ndcg_scores.append(r_ndcg)
            print("  Reranked  -> RR@{}: {:.3f} | NDCG@{}: {:.3f} | {:.2f}s".format(
                args.k_mrr, r_rr, args.k_ndcg, r_ndcg, reranked_time))

            per_query_results.append({
                "query_id":      query_id,
                "query_text":    query_text,
                "baseline_rr":   round(b_rr, 6),
                "baseline_ndcg": round(b_ndcg, 6),
                "reranked_rr":   round(r_rr, 6),
                "reranked_ndcg": round(r_ndcg, 6),
                "relevant_docs": list(relevant),
            })

    # Final metrics
    def mean(scores):
        return sum(scores) / len(scores) if scores else 0.0

    final_metrics = {
        "baseline": {
            "mrr_at_5":   round(mean(baseline_rr_scores), 6),
            "ndcg_at_10": round(mean(baseline_ndcg_scores), 6),
        },
        "reranked": {
            "mrr_at_5":   round(mean(reranked_rr_scores), 6),
            "ndcg_at_10": round(mean(reranked_ndcg_scores), 6),
        },
    }

    # Save results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    detailed_path = RESULTS_PATH.parent / "evaluation_detailed.json"
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump({"summary": final_metrics, "per_query": per_query_results}, f, indent=2)

    # Print summary
    mrr_b  = final_metrics["baseline"]["mrr_at_5"]
    mrr_r  = final_metrics["reranked"]["mrr_at_5"]
    ndcg_b = final_metrics["baseline"]["ndcg_at_10"]
    ndcg_r = final_metrics["reranked"]["ndcg_at_10"]

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print("  {:<20} {:>12} {:>12} {:>15}".format("Metric", "Baseline", "Re-ranked", "Improvement"))
    print("-" * 60)
    print("  {:<20} {:>12.4f} {:>12.4f} {:>+14.4f}".format("MRR@5", mrr_b, mrr_r, mrr_r - mrr_b))
    print("  {:<20} {:>12.4f} {:>12.4f} {:>+14.4f}".format("NDCG@10", ndcg_b, ndcg_r, ndcg_r - ndcg_b))
    print("=" * 60)
    print("  Results saved to {}".format(RESULTS_PATH))
    print("=" * 60)


if __name__ == "__main__":
    main()