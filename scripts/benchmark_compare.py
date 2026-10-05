#!/usr/bin/env python3
"""Latency + hit-rate benchmark: local hybrid pipeline vs Vertex AI RAG Engine.

Phases per engine:
  retrieval-only  query embedding + search (no LLM)
  end-to-end      retrieve + gemini-2.5-flash answer

Protocol per handoff: 2 warmup runs (first question), then per question
  - retrieval-only: 10 timed runs
  - end-to-end:      3 timed runs (each is a paid Gemini call)

Outputs:
  reports/latency_comparison.json
  reports/latency_comparison.md

Run:
  python scripts/benchmark_compare.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DATABASE_PATH", "data/processed/vectors_large.db")
os.environ.setdefault("EMBEDDING_BACKEND", "google")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import warnings

warnings.filterwarnings("ignore")

from src.config import settings  # noqa: E402
from src.embeddings.factory import build_embedder  # noqa: E402
from src.llm.gemini_chat import GeminiChat  # noqa: E402
from src.retrieval.rag_engine_retriever import RagEngineRetriever  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.vectordb.vector_store import VectorStore  # noqa: E402

import agentplatform  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types as gtypes  # noqa: E402

QUESTIONS_PATH = REPO_ROOT / "benchmarks" / "queries.jsonl"
REPORTS_DIR = REPO_ROOT / "reports"

TOP_K = 5
WARMUP_RUNS = 2
RETRIEVAL_RUNS = 10
E2E_RUNS = 3

PROMPT = (
    "You are a legal research assistant. Answer the question using ONLY the "
    "context below. If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def load_questions() -> list[dict]:
    questions = []
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(json.loads(line))
    return questions


def fmt_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, start=1):
        name = r["metadata"].get("filename") or "unknown"
        parts.append(f"[{i}] ({name})\n{r['text']}")
    return "\n\n".join(parts)


def filenames_of(results: list[dict]) -> list[str]:
    return [
        (r["metadata"].get("filename") or "").lower()
        for r in results
    ]


def hit_rates(results: list[dict], expected: list[str]) -> dict:
    names = filenames_of(results)
    exp = [e.lower() for e in expected]
    hit1 = bool(names) and any(names[0] == e or e in names[0] for e in exp)
    hit5 = any(any(e in n for n in names) for e in exp)
    return {"hit_at_1": hit1, "hit_at_5": hit5}


def stats_of(values_ms: list[float]) -> dict:
    if not values_ms:
        return {}
    s = sorted(values_ms)
    return {
        "p50_ms": round(statistics.median(s), 1),
        "p95_ms": round(s[max(0, int(len(s) * 0.95) - 1)], 1),
        "mean_ms": round(statistics.fmean(s), 1),
        "min_ms": round(s[0], 1),
        "max_ms": round(s[-1], 1),
    }


def bench_engine_retrieval(name: str, fn, questions: list[dict]) -> dict:
    per_question = []
    warmed = False
    for qi, q in enumerate(questions, start=1):
        if not warmed:
            for _ in range(WARMUP_RUNS):
                fn(q["question"])
            warmed = True
            print(f"  [{name}] warmup done", flush=True)

        hits = None
        times = []
        for run in range(RETRIEVAL_RUNS):
            t0 = time.perf_counter()
            results = fn(q["question"])
            times.append((time.perf_counter() - t0) * 1000)
            if run == 0:
                hits = hit_rates(results, q.get("expected_filenames", []))
        mean_ms = statistics.fmean(times)
        print(
            f"  [{name}] q{qi}/{len(questions)} retrieval p50={statistics.median(times):.0f}ms "
            f"hit@5={hits['hit_at_5']}",
            flush=True,
        )
        per_question.append(
            {"id": q["id"], "question": q["question"], "category": q.get("category"),
             "hit": hits, "latency_ms": [round(t, 1) for t in times]}
        )
    all_times = [t for pq in per_question for t in pq["latency_ms"]]
    hit5 = statistics.fmean([1.0 if pq["hit"]["hit_at_5"] else 0.0 for pq in per_question])
    hit1 = statistics.fmean([1.0 if pq["hit"]["hit_at_1"] else 0.0 for pq in per_question])
    return {
        "latency": stats_of(all_times),
        "hit_rate_at_5": round(hit5, 3),
        "hit_rate_at_1": round(hit1, 3),
        "per_question": per_question,
    }


def _with_429_backoff(fn, *args, **kwargs):
    """Call fn, retrying Vertex 429 RESOURCE_EXHAUSTED with growing backoff.

    Free/trial Gemini quotas are tight; the E2E loop bursts 3 runs per
    question. Backoff 30s/60s/120s/240s/480s, then give up.
    """
    delays = (30, 60, 120, 240, 480)
    for attempt, delay in enumerate((0,) + delays):
        if delay:
            print(f"    429 backoff: waiting {delay}s (attempt {attempt}/{len(delays)})",
                  flush=True)
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if "429" not in str(exc) or attempt == len(delays):
                raise
    raise AssertionError("unreachable")


def bench_engine_e2e(name: str, fn, questions: list[dict], on_progress=None) -> dict:
    per_question = []
    for qi, q in enumerate(questions, start=1):
        times = []
        answer_preview = ""
        for run in range(WARMUP_RUNS + E2E_RUNS):
            t0 = time.perf_counter()
            answer = _with_429_backoff(fn, q["question"])
            times.append((time.perf_counter() - t0) * 1000)
            if run == WARMUP_RUNS:
                answer_preview = answer[:150]
        timed = times[WARMUP_RUNS:]
        print(
            f"  [{name}] q{qi}/{len(questions)} e2e p50={statistics.median(timed):.0f}ms",
            flush=True,
        )
        per_question.append(
            {"id": q["id"], "latency_ms": [round(t, 1) for t in timed],
             "answer_preview": answer_preview}
        )
        if on_progress:
            on_progress({"latency": stats_of(
                [t for pq in per_question for t in pq["latency_ms"]]),
                "per_question": per_question})
        if qi < len(questions):
            time.sleep(15)  # pace below per-minute Gemini quotas
    all_times = [t for pq in per_question for t in pq["latency_ms"]]
    return {"latency": stats_of(all_times), "per_question": per_question}


def bench_stream_e2e(name: str, pipeline, questions: list[dict], on_progress=None) -> dict:
    """E2E via the production streaming path (RAGPipeline.stream_answer).

    Measures, per run: total answer latency and TTFB (time to first
    token - retrieval + prompt build + first LLM token). TTFB is the
    number a streaming UI actually feels. Uses the production
    token-budgeted prompt and the pipeline's own stream-start 429 retry.
    """
    per_question = []
    for qi, q in enumerate(questions, start=1):
        times, ttfbs, answer_preview = [], [], ""
        for run in range(WARMUP_RUNS + E2E_RUNS):
            t0 = time.perf_counter()
            stream, chunks, sources = pipeline.stream_answer(q["question"])
            ttfb_ms = None
            parts = []
            for token in stream:
                if ttfb_ms is None:
                    ttfb_ms = (time.perf_counter() - t0) * 1000
                parts.append(token)
            times.append((time.perf_counter() - t0) * 1000)
            if ttfb_ms is not None:
                ttfbs.append(ttfb_ms)
            if run == WARMUP_RUNS:
                answer_preview = "".join(parts)[:150]
        timed, ttfb_timed = times[WARMUP_RUNS:], ttfbs[WARMUP_RUNS:]
        print(
            f"  [{name}] q{qi}/{len(questions)} stream-e2e "
            f"p50={statistics.median(timed):.0f}ms ttfb_p50={statistics.median(ttfb_timed):.0f}ms",
            flush=True,
        )
        per_question.append(
            {"id": q["id"], "latency_ms": [round(t, 1) for t in timed],
             "ttfb_ms": [round(t, 1) for t in ttfb_timed],
             "answer_preview": answer_preview}
        )
        if on_progress:
            on_progress({"latency": stats_of(
                [t for pq in per_question for t in pq["latency_ms"]]),
                "per_question": per_question})
        if qi < len(questions):
            time.sleep(15)  # pace below per-minute Gemini quotas
    all_times = [t for pq in per_question for t in pq["latency_ms"]]
    all_ttfb = [t for pq in per_question for t in pq["ttfb_ms"]]
    return {"latency": stats_of(all_times), "ttfb": stats_of(all_ttfb),
            "per_question": per_question}


def main() -> None:
    global RETRIEVAL_RUNS, E2E_RUNS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-runs", type=int, default=RETRIEVAL_RUNS)
    parser.add_argument("--e2e-runs", type=int, default=E2E_RUNS)
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument(
        "--phase", choices=["all", "retrieval", "retrieval-rag", "retrieval-local",
                            "e2e", "stream-e2e"],
        default="all",
        help="'retrieval' saves retrieval metrics; 'retrieval-local'/'retrieval-rag' "
             "re-measure one side into a saved report; 'e2e' merges E2E metrics into it; "
             "'stream-e2e' measures the local production streaming path (latency + TTFB)",
    )
    parser.add_argument(
        "--e2e-questions", type=int, default=0,
        help="limit the E2E phase to the first N questions (0 = all)",
    )
    args = parser.parse_args()
    RETRIEVAL_RUNS = args.retrieval_runs
    E2E_RUNS = args.e2e_runs

    questions = load_questions()
    e2e_questions = questions[: args.e2e_questions] if args.e2e_questions else questions
    run_retrieval = args.phase in ("all", "retrieval")
    run_retrieval_local = args.phase in ("all", "retrieval", "retrieval-local")
    run_retrieval_rag = args.phase in ("all", "retrieval", "retrieval-rag")
    run_e2e = args.phase in ("all", "e2e") and not args.skip_e2e
    run_stream_e2e = args.phase in ("all", "stream-e2e")
    json_path = REPORTS_DIR / "latency_comparison.json"

    saved = None
    if args.phase in ("retrieval-rag", "retrieval-local", "e2e"):
        if not json_path.exists():
            raise SystemExit("No saved report - run --phase retrieval first")
        saved = json.loads(json_path.read_text(encoding="utf-8"))

    report = saved or {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "gcp_project": settings.gcp_project_id,
            "gcp_location": settings.gcp_location,
            "rag_corpus": settings.rag_corpus_name,
            "local_embedding": "text-embedding-004 (768d)",
            "rag_embedding": "text-embedding-005 (serverless default)",
            "llm": settings.gemini_model,
            "top_k": TOP_K,
            "retrieval_runs_per_question": RETRIEVAL_RUNS,
            "e2e_runs_per_question": E2E_RUNS,
            "e2e_questions": args.e2e_questions or None,
            "warmup_runs": WARMUP_RUNS,
            "questions": len(questions),
        },
        "local": {"retrieval": None, "e2e": None},
        "rag_engine": {"retrieval": None, "e2e": None},
    }

    # --- Local engine ---
    if run_retrieval_local or run_e2e or run_stream_e2e:
        print("== LOCAL (hybrid sqlite-vec + FTS5, text-embedding-004 queries) ==",
              flush=True)
        vs = VectorStore(database_path=str(settings.database_path), default_top_k=TOP_K)
        local_retriever = Retriever(
            vector_store=vs,
            embedder=build_embedder(settings),
            # Use the system's configured threshold, not the class default -
            # a benchmark artifact here would distort the E2E comparison.
            default_similarity_threshold=settings.similarity_threshold,
        )
        local_llm = GeminiChat(
            project_id=settings.gcp_project_id,
            location=settings.gcp_location,
            model=settings.gemini_model,
        )

        def local_e2e(question: str) -> str:
            results = local_retriever.retrieve_with_scores(question, top_k=TOP_K)
            return local_llm.generate(
                PROMPT.format(context=fmt_context(results), question=question)
            )

        if run_retrieval_local:
            t0 = time.perf_counter()
            report["local"]["retrieval"] = bench_engine_retrieval(
                "local", local_retriever.retrieve_with_scores, questions
            )
            print(f"  local retrieval phase: {time.perf_counter()-t0:.0f}s", flush=True)

        local_e2e_res = None
        if run_e2e and not report["local"].get("e2e"):
            t0 = time.perf_counter()
            local_e2e_res = bench_engine_e2e("local", local_e2e, e2e_questions)
            print(f"  local e2e phase: {time.perf_counter()-t0:.0f}s", flush=True)
        report["local"]["e2e"] = local_e2e_res or report["local"].get("e2e")

        if run_stream_e2e:
            from src.rag.rag_pipeline import RAGPipeline

            pipeline = RAGPipeline(
                retriever=local_retriever,
                llm=local_llm,
                top_k=TOP_K,
                similarity_threshold=settings.similarity_threshold,
            )
            t0 = time.perf_counter()
            report["local"]["e2e_stream"] = bench_stream_e2e(
                "local", pipeline, e2e_questions
            )
            print(f"  local stream-e2e phase: {time.perf_counter()-t0:.0f}s", flush=True)
        vs.close()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("  (saved after local phase)", flush=True)

    # --- RAG Engine ---
    if run_retrieval_rag or run_e2e:
        print("== RAG ENGINE (serverless, us-central1, text-embedding-005) ==",
              flush=True)
        rag_retriever = RagEngineRetriever(default_top_k=TOP_K)
        rag_client = genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )

        def rag_e2e(question: str) -> str:
            resp = rag_client.models.generate_content(
                model=settings.gemini_model,
                contents=question,
                config=gtypes.GenerateContentConfig(
                    tools=[gtypes.Tool(
                        retrieval=gtypes.Retrieval(
                            vertex_rag_store=gtypes.VertexRagStore(
                                rag_corpora=[settings.rag_corpus_name],
                                similarity_top_k=TOP_K,
                            )
                        )
                    )],
                ),
            )
            return resp.text or ""

        if run_retrieval_rag:
            t0 = time.perf_counter()
            report["rag_engine"]["retrieval"] = bench_engine_retrieval(
                "rag", rag_retriever.retrieve_with_scores, questions
            )
            print(f"  rag retrieval phase: {time.perf_counter()-t0:.0f}s", flush=True)

        rag_e2e_res = None
        if run_e2e and not report["rag_engine"].get("e2e"):
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)

            def _save_partial(partial: dict) -> None:
                report["rag_engine"]["e2e_partial"] = partial
                json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            t0 = time.perf_counter()
            rag_e2e_res = bench_engine_e2e(
                "rag", rag_e2e, e2e_questions, on_progress=_save_partial
            )
            print(f"  rag e2e phase: {time.perf_counter()-t0:.0f}s", flush=True)
            report["rag_engine"]["e2e"] = rag_e2e_res
            report["rag_engine"].pop("e2e_partial", None)
            json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["rag_engine"]["e2e"] = rag_e2e_res or report["rag_engine"].get("e2e")

    # --- Report ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = render_md(report)
    (REPORTS_DIR / "latency_comparison.md").write_text(md, encoding="utf-8")
    print(f"\nWrote {json_path} and reports/latency_comparison.md", flush=True)


def render_md(r: dict) -> str:
    m = r["meta"]
    lines = [
        "# Local hybrid RAG vs Vertex AI RAG Engine - latency benchmark",
        "",
        f"*{m['timestamp_utc']} | project `{m['gcp_project']}` @ `{m['gcp_location']}` | "
        f"{m['questions']} questions | top_k={m['top_k']} | "
        f"{m['retrieval_runs_per_question']} timed retrieval runs, {m['e2e_runs_per_question']} timed E2E runs per question*",
        "",
        "## Headline numbers",
        "",
        "| Metric | Local (hybrid, sqlite-vec+FTS5) | RAG Engine (serverless) |",
        "|---|---|---|",
    ]
    lr, rr = r["local"]["retrieval"], r["rag_engine"]["retrieval"]
    if lr and rr:
        lines.append(f"| Retrieval p50 | {lr['latency']['p50_ms']} ms | {rr['latency']['p50_ms']} ms |")
        lines.append(f"| Retrieval p95 | {lr['latency']['p95_ms']} ms | {rr['latency']['p95_ms']} ms |")
        lines.append(f"| Hit-rate@5 | {lr['hit_rate_at_5']*100:.0f}% | {rr['hit_rate_at_5']*100:.0f}% |")
        lines.append(f"| Hit-rate@1 | {lr['hit_rate_at_1']*100:.0f}% | {rr['hit_rate_at_1']*100:.0f}% |")
    if r["local"].get("e2e_stream"):
        s = r["local"]["e2e_stream"]
        lines.append(f"| Stream E2E p50 | {s['latency']['p50_ms']} ms | - |")
        lines.append(f"| Stream TTFB p50 | {s['ttfb']['p50_ms']} ms | - |")
    if r["local"]["e2e"] and r["rag_engine"]["e2e"]:
        le, re_ = r["local"]["e2e"]["latency"], r["rag_engine"]["e2e"]["latency"]
        lines.append(f"| E2E p50 | {le['p50_ms']} ms | {re_['p50_ms']} ms |")
        lines.append(f"| E2E p95 | {le['p95_ms']} ms | {re_['p95_ms']} ms |")
    lines += [
        "",
        "## Methodology notes",
        "",
        "- Local: query embedded with text-embedding-004 (768-d, same as stored vectors), "
        "hybrid sqlite-vec ANN + FTS5 keyword search on the 100 MB corpus (50,318 chunks).",
        "- RAG Engine: serverless corpus in us-central1 (serverless is us-central1-only); "
        "query embedded + searched server-side with its text-embedding-005 default "
        "(004 explicit override is ignored by serverless mode).",
        "- E2E uses gemini-2.5-flash for both engines; the RAG side retrieves inside the "
        "generate call via the built-in VertexRagStore grounding tool.",
        "- Benchmarked from India: us-central1 queries carry ~200 ms network RTT that is "
        "included in all numbers above, as it would be in production.",
        "- Local similarity scores are L2-based (monotonic with cosine for unit vectors); "
        "RAG Engine reports its own distance/score - hit-rate uses filenames only.",
        "",
        "## Per-question detail",
        "",
        "| # | Category | Local p50 | RAG p50 | Local hit@5 | RAG hit@5 |",
        "|---|---|---|---|---|---|",
    ]
    if lr and rr:
        for lpq, rpq in zip(lr["per_question"], rr["per_question"]):
            lp50 = statistics.median(lpq["latency_ms"])
            rp50 = statistics.median(rpq["latency_ms"])
            lines.append(
                f"| {lpq['id']} | {lpq['category']} | {lp50:.0f} ms | {rp50:.0f} ms | "
                f"{'Y' if lpq['hit']['hit_at_5'] else 'N'} | {'Y' if rpq['hit']['hit_at_5'] else 'N'} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()