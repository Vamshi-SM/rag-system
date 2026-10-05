#!/usr/bin/env python3
"""Migrate vectors_large.db (SQLite/sqlite-vec) into a local pgvector
container and time SEARCH-ONLY latency for the benchmark questions
across: sqlite-vec brute force, pgvector exact scan, pgvector HNSW.

Search-only timing excludes query embedding (identical for all stores),
so the delta is pure database performance.
"""

from __future__ import annotations

import json
import os
import statistics
import struct
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("EMBEDDING_BACKEND", "google")

import numpy as np

from src.config import settings
from src.vectordb.vector_store import VectorStore

DB_URL = "postgresql://postgres:postgres@localhost:5434/postgres"
SQLITE_DB = str(REPO / "data" / "processed" / "vectors_large.db")
QUESTIONS = [
    json.loads(line)
    for line in (REPO / "benchmarks" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def main() -> None:
    import psycopg
    from pgvector.psycopg import register_vector

    # ---- 1. Load everything from SQLite ----
    print("Reading SQLite...", flush=True)
    import sqlite3

    import sqlite_vec

    con = sqlite3.connect(SQLITE_DB)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    rows = con.execute(
        "SELECT c.id, c.chunk_id, c.document_id, c.text, c.metadata, v.embedding "
        "FROM chunks c JOIN chunks_vec v ON v.rowid = c.rowid"
    ).fetchall()
    con.close()
    print(f"  {len(rows)} chunks", flush=True)

    # ---- 2. Load into pgvector (no index during load) ----
    print("Loading into pgvector...", flush=True)
    conn = psycopg.connect(DB_URL, autocommit=True)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)
    cur.execute("DROP TABLE IF EXISTS chunks;")
    cur.execute("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            document_id TEXT,
            text TEXT NOT NULL,
            metadata JSONB,
            embedding VECTOR(768)
        );
    """)
    t0 = time.perf_counter()
    BATCH = 500
    for i in range(0, len(rows), BATCH):
        batch = rows[i : i + BATCH]
        data = []
        for r in batch:
            vec = np.frombuffer(r[5], dtype="<f4")
            vec_text = "[" + ",".join(f"{v:.6g}" for v in vec) + "]"
            data.append((r[0], r[1], r[2], r[3], r[4], vec_text))
        cur.executemany(
            "INSERT INTO chunks (id, chunk_id, document_id, text, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)",
            data,
        )
    load_s = time.perf_counter() - t0
    print(f"  load: {load_s:.1f}s", flush=True)

    # ---- 3. Build HNSW index (timed) ----
    print("Building HNSW index...", flush=True)
    t0 = time.perf_counter()
    cur.execute(
        "CREATE INDEX chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64);"
    )
    hnsw_s = time.perf_counter() - t0
    print(f"  HNSW build: {hnsw_s:.1f}s", flush=True)

    # ---- 4. Embed the 25 questions once (shared by all stores) ----
    from src.embeddings.factory import build_embedder
    embedder = build_embedder(settings)
    print("Embedding 25 questions...", flush=True)
    qvecs = []
    for q in QUESTIONS:
        qvecs.append(embedder.embed(q["question"]))

    # ---- 5. Search-only timings ----
    RUNS = 10
    sqlite_store = VectorStore(database_path=SQLITE_DB, default_top_k=5)

    def time_store(fn) -> list[float]:
        times = []
        for qv in qvecs:
            fn(qv)  # warmup
            for _ in range(RUNS):
                t0 = time.perf_counter()
                fn(qv)
                times.append((time.perf_counter() - t0) * 1000)
        return times

    print("Timing sqlite-vec brute force...", flush=True)
    t_sqlite = time_store(
        lambda qv: sqlite_store.similarity_search(qv, top_k=5)
    )

    def vec_text(qv) -> str:
        return "[" + ",".join(f"{v:.6g}" for v in qv) + "]"

    def pg_hnsw(qv):
        cur.execute(
            "SELECT id, chunk_id, document_id, text, metadata, "
            "embedding <=> %s::vector AS distance "
            "FROM chunks ORDER BY distance LIMIT 5",
            (vec_text(qv),),
        )
        return cur.fetchall()

    def pg_exact(qv):
        cur.execute("SET enable_indexscan = off; SET enable_bitmapscan = off;")
        try:
            cur.execute(
                "SELECT id, chunk_id, document_id, text, metadata, "
                "embedding <=> %s::vector AS distance "
                "FROM chunks ORDER BY distance LIMIT 5",
                (vec_text(qv),),
            )
            return cur.fetchall()
        finally:
            cur.execute("SET enable_indexscan = on; SET enable_bitmapscan = on;")

    print("Timing pgvector HNSW...", flush=True)
    t_hnsw = time_store(pg_hnsw)

    print("Timing pgvector exact scan...", flush=True)
    t_exact = time_store(pg_exact)

    # server-side actual execution time (EXPLAIN ANALYZE), 10 samples
    srv_times = []
    for qv in qvecs[:10]:
        cur.execute(
            "EXPLAIN ANALYZE SELECT id FROM chunks "
            "ORDER BY embedding <=> %s::vector LIMIT 5",
            (vec_text(qv),),
        )
        for r in cur.fetchall():
            if "Execution Time" in r[0]:
                srv_times.append(float(r[0].split(":")[1].replace("ms", "").strip()))
    print(f"HNSW server-side p50={statistics.median(srv_times):.2f}ms", flush=True)

    def stats(name, times):
        s = sorted(times)
        print(
            f"{name:24s} p50={statistics.median(s):7.1f}ms  "
            f"p95={s[int(len(s)*0.95)-1]:7.1f}ms  min={s[0]:6.1f}ms  max={s[-1]:7.1f}ms",
            flush=True,
        )
        return {
            "p50_ms": round(statistics.median(s), 1),
            "p95_ms": round(s[int(len(s) * 0.95) - 1], 1),
            "min_ms": round(s[0], 1),
            "max_ms": round(s[-1], 1),
            "n": len(s),
        }

    print("\n=== SEARCH-ONLY latency (250 timed queries each, 25 questions x 10) ===")
    results = {
        "meta": {
            "vectors": len(rows),
            "dim": 768,
            "questions": len(QUESTIONS),
            "runs_per_question": RUNS,
            "hnsw_build_seconds": round(hnsw_s, 1),
            "pg_load_seconds": round(load_s, 1),
            "hnsw_params": "m=16, ef_construction=64",
        },
        "sqlite_vec_bruteforce": stats("sqlite-vec brute force", t_sqlite),
        "pgvector_hnsw": stats("pgvector HNSW (client, docker)", t_hnsw),
        "pgvector_exact": stats("pgvector exact (client, docker)", t_exact),
        "pgvector_hnsw_server_side_ms": {
            "p50": round(statistics.median(srv_times), 2),
            "min": round(min(srv_times), 2),
            "max": round(max(srv_times), 2),
        },
    }

    # correctness spot-check: same top-1 id?
    cur.execute(
        "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 5",
        (vec_text(qvecs[0]),),
    )
    pg_top1 = cur.fetchone()[0]
    sq_top1 = sqlite_store.similarity_search(qvecs[0], top_k=1)[0]["id"]
    results["top1_agreement_q1"] = bool(pg_top1 == sq_top1)
    print(f"top-1 agreement (q1): {pg_top1 == sq_top1} ({pg_top1} vs {sq_top1})", flush=True)

    out = REPO / "reports" / "vectordb_comparison.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    sqlite_store.close()
    conn.close()


if __name__ == "__main__":
    main()