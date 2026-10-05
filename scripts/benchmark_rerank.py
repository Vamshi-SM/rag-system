#!/usr/bin/env python3
"""Retrieval benchmark on the full local corpus (50k chunks), with or
without the cross-encoder reranker.

Same protocol as the saved baselines (25 questions x 10 timed runs, top-5,
threshold from settings). Measures hit-rate and latency; writes
reports/retrieval_large_db.json (no reranker) or reports/reranker_benchmark.json.

Model download happens once (HF cache); every query after that is local CPU.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reranker", action="store_true", help="enable cross-encoder reranker")
    p.add_argument(
        "--query-task",
        default="RETRIEVAL_QUERY",
        choices=["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"],
        help="task_type used to embed queries (A/B experiment)",
    )
    p.add_argument(
        "--candidate-k", type=int, default=None,
        help="override how many candidates hybrid retrieval pulls before rerank",
    )
    return p.parse_args()


class TaskTypeEmbedder:
    """Wraps GoogleEmbedding to embed queries with an explicit task type.

    The DB-side chunk embeddings are RETRIEVAL_DOCUMENT; this controls only
    the query side for A/B ranking experiments.
    """

    def __init__(self, inner, task_type: str):
        self._inner = inner
        self._task = task_type

    def embed(self, text: str):
        return self._inner._embed_with_retry(
            contents=text, task_type=self._task
        ).embeddings[0].values

    def __getattr__(self, name):
        return getattr(self._inner, name)

from src.config import settings  # noqa: E402
from src.embeddings.factory import build_embedder  # noqa: E402
from src.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.vectordb.vector_store import VectorStore  # noqa: E402

QUESTIONS = [
    json.loads(line)
    for line in (REPO / "benchmarks" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
WARMUP, RUNS, TOP_K = 2, 10, 5


def hit_rates(results, expected):
    names = [(r["metadata"].get("filename") or "").lower() for r in results]
    exp = [e.lower() for e in expected]
    hit1 = bool(names) and any(names[0] == e or e in names[0] for e in exp)
    hit5 = any(any(e in n for n in names) for e in exp)
    return {"hit_at_1": hit1, "hit_at_5": hit5}


def main() -> None:
    args = parse_args()
    store = VectorStore(database_path="unused", default_top_k=TOP_K)
    reranker = None
    if args.reranker:
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
            top_n=settings.reranker_top_n,
            max_chars=int(os.getenv("RERANKER_MAX_CHARS", "1200")),
        )
    retriever = Retriever(
        vector_store=store,
        embedder=TaskTypeEmbedder(build_embedder(settings), args.query_task),
        default_similarity_threshold=settings.similarity_threshold,
        reranker=reranker,
        rerank_top_n=settings.reranker_top_n,
    )

    per_question = []
    warmed = False
    for qi, q in enumerate(QUESTIONS, start=1):
        if not warmed:
            for _ in range(WARMUP):
                retriever.retrieve_with_scores(q["question"], top_k=TOP_K)
            warmed = True
            print("  warmup done (model loaded)", flush=True)
        times, hits = [], None
        for run in range(RUNS):
            t0 = time.perf_counter()
            results = retriever.retrieve_with_scores(
                q["question"], top_k=TOP_K, candidate_k_override=args.candidate_k
            )
            times.append((time.perf_counter() - t0) * 1000)
            if run == 0:
                hits = hit_rates(results, q.get("expected_filenames", []))
        print(
            f"  q{qi}/{len(QUESTIONS)} p50={statistics.median(times):.0f}ms "
            f"hit@5={hits['hit_at_5']}", flush=True,
        )
        per_question.append({
            "id": q["id"], "question": q["question"],
            "hit": hits, "latency_ms": [round(t, 1) for t in times],
        })

    all_times = [t for p in per_question for t in p["latency_ms"]]
    s = sorted(all_times)
    hit5 = statistics.fmean([1.0 if p["hit"]["hit_at_5"] else 0.0 for p in per_question])
    hit1 = statistics.fmean([1.0 if p["hit"]["hit_at_1"] else 0.0 for p in per_question])
    summary = {
        "p50_ms": round(statistics.median(s), 1),
        "p95_ms": round(s[int(len(s) * 0.95) - 1], 1),
        "mean_ms": round(statistics.fmean(s), 1),
        "hit_rate_at_5": round(hit5, 3),
        "hit_rate_at_1": round(hit1, 3),
    }
    label = f"{'WITH reranker' if args.reranker else 'NO reranker'} [{args.query_task}]"
    print(f"\n{label}: p50={summary['p50_ms']}ms p95={summary['p95_ms']}ms "
          f"hit@5={summary['hit_rate_at_5']*100:.0f}% hit@1={summary['hit_rate_at_1']*100:.0f}%",
          flush=True)

    misses = [p["id"] for p in per_question if not p["hit"]["hit_at_5"]]
    print(f"hit@5 misses: {misses or 'none'}", flush=True)

    out = {
        "meta": {
            "backend": "postgres/pgvector local container",
            "query_task_type": args.query_task,
            "reranker": settings.reranker_model if args.reranker else None,
            "rerank_top_n": settings.reranker_top_n if args.reranker else None,
            "rerank_max_chars": int(os.getenv("RERANKER_MAX_CHARS", "1200")) if args.reranker else None,
            "candidate_k": args.candidate_k,
            "similarity_threshold": settings.similarity_threshold,
            "questions": len(QUESTIONS), "runs": RUNS, "top_k": TOP_K,
        },
        "retrieval_with_reranker" if args.reranker else "retrieval_no_reranker": summary,
        "per_question": per_question,
        "baselines": {
            "rag_engine": {"p50_ms": 846.0, "p95_ms": 1376.5, "hit5": 1.0},
        },
    }
    suffix = args.query_task.replace("RETRIEVAL_", "").lower()
    if args.candidate_k:
        suffix += f"_c{args.candidate_k}"
    out_name = (
        f"reranker_benchmark_{suffix}.json" if args.reranker
        else f"retrieval_large_db_{suffix}.json"
    )
    (REPO / "reports" / out_name).write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"wrote reports/{out_name}", flush=True)
    store.close()


if __name__ == "__main__":
    main()