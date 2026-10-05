#!/usr/bin/env python3
"""Benchmark the local retrieval pipeline on the pgvector backend.

Runs the SAME VectorStore/Retriever code path with DATABASE_BACKEND=postgres
against the migrated corpus in the local pgvector container, and compares:

  1. top-5 equivalence vs the sqlite-vec store (all 25 questions)
  2. full retrieval latency (embed + search + RRF fuse), 2 warmup + 10
     timed runs per question - comparable with the saved sqlite (942 ms)
     and RAG Engine (846 ms) metrics in reports/latency_comparison.json

No LLM calls: with identical top-5 chunks the E2E answer is identical by
construction, so E2E is not re-measured.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ["DATABASE_BACKEND"] = "postgres"
os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5434/postgres"
os.environ.setdefault("EMBEDDING_BACKEND", "google")
os.environ.setdefault("DATABASE_PATH", "data/processed/vectors_large.db")

from src.config import settings  # noqa: E402
from src.embeddings.factory import build_embedder  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.vectordb.vector_store import VectorStore  # noqa: E402

QUESTIONS = [
    json.loads(line)
    for line in (REPO / "benchmarks" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
WARMUP, RUNS = 2, 10


def main() -> None:
    print(f"backend={settings.database_backend} url={settings.database_url}", flush=True)

    print("Building sqlite retriever (equivalence reference)...", flush=True)
    sq_store = VectorStore(database_path=str(settings.database_path), default_top_k=5)
    sq_retr = Retriever(
        vector_store=sq_store,
        embedder=build_embedder(settings),
        default_similarity_threshold=settings.similarity_threshold,
    )

    print("Building pgvector retriever...", flush=True)
    pg_store = VectorStore(database_path="unused", default_top_k=5)
    pg_retr = Retriever(
        vector_store=pg_store,
        embedder=build_embedder(settings),
        default_similarity_threshold=settings.similarity_threshold,
    )

    # ---- 1. top-5 equivalence per question ----
    agree5, agree_top1 = 0, 0
    for q in QUESTIONS:
        sq = sq_retr.retrieve_with_scores(q["question"], top_k=5)
        pg = pg_retr.retrieve_with_scores(q["question"], top_k=5)
        sq_ids = [r["metadata"].get("chunk_id") for r in sq]
        pg_ids = [r["metadata"].get("chunk_id") for r in pg]
        if sq_ids and pg_ids and sq_ids[0] == pg_ids[0]:
            agree_top1 += 1
        if set(sq_ids) == set(pg_ids):
            agree5 += 1
    n = len(QUESTIONS)
    print(f"top-5 set agreement: {agree5}/{n} | top-1 agreement: {agree_top1}/{n}",
          flush=True)

    # ---- 2. retrieval-only latency on pgvector ----
    times = []
    warmed = False
    for qi, q in enumerate(QUESTIONS, start=1):
        if not warmed:
            for _ in range(WARMUP):
                pg_retr.retrieve_with_scores(q["question"], top_k=5)
            warmed = True
        for _ in range(RUNS):
            t0 = time.perf_counter()
            pg_retr.retrieve_with_scores(q["question"], top_k=5)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  [pg] q{qi}/{n} done", flush=True)
    s = sorted(times)
    summary = {
        "p50_ms": round(statistics.median(s), 1),
        "p95_ms": round(s[int(len(s) * 0.95) - 1], 1),
        "mean_ms": round(statistics.fmean(s), 1),
        "min_ms": round(s[0], 1),
        "max_ms": round(s[-1], 1),
    }
    print(f"pgvector retrieval p50={summary['p50_ms']}ms p95={summary['p95_ms']}ms",
          flush=True)

    out = {
        "meta": {
            "backend": "postgres/pgvector (local docker container, port 5434)",
            "vectors": 50318,
            "dim": 768,
            "questions": n,
            "warmup": WARMUP,
            "runs_per_question": RUNS,
            "top_k": 5,
            "similarity_threshold": settings.similarity_threshold,
            "embedding": "text-embedding-004 (same as stored vectors)",
        },
        "equivalence": {"top5_set": f"{agree5}/{n}", "top1": f"{agree_top1}/{n}"},
        "retrieval_latency": summary,
        "references": {
            "local_sqlite_retrieval_p50_ms": 942.5,
            "rag_engine_retrieval_p50_ms": 846.0,
        },
    }
    (REPO / "reports" / "pgvector_vs.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("wrote reports/pgvector_vs.json", flush=True)
    sq_store.close()
    pg_store.close()


if __name__ == "__main__":
    main()