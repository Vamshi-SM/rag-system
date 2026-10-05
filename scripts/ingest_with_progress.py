#!/usr/bin/env python3
"""Ingestion runner with live progress for the 100 MB corpus.

Unlike ``IngestionService.run`` (which embeds EVERYTHING and then does one
giant insert), this runner works in incremental waves:

    chunk once (free, local CPU) -> per wave of ~2048 chunks:
        embed -> insert_many (single transaction) -> register its documents

A crash at wave N loses at most one wave; a re-run skips every registered
document via the checksum registry, and deterministic chunk ids make the
re-embed an idempotent upsert. Safe against machine sleep, CLI harness
kills, and API outages mid-run.

Environment is pinned here (before src imports read settings):
  EMBEDDING_BACKEND=google (Vertex AI text-embedding-004)
  EMBEDDING_BATCH_SIZE=27, EMBEDDING_TOKEN_BUDGET=19000
      (text-embedding-004 caps a request at 20k TOTAL tokens -> ~27 x
       700-token chunks; larger batches are rejected with a 400)
  EMBEDDING_CONCURRENCY=8
  DATABASE_PATH=data/processed/vectors_large.db

Requires Application Default Credentials for the target GCP project:
  gcloud auth application-default login
  gcloud auth application-default set-quota-project <project>

Run:
  python scripts/ingest_with_progress.py            # resume-safe
  python scripts/ingest_with_progress.py --fresh    # delete DB first
  python scripts/ingest_with_progress.py --limit 4  # smoke test (real docs)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("EMBEDDING_BACKEND", "google")
os.environ.setdefault("EMBEDDING_BATCH_SIZE", "27")
os.environ.setdefault("EMBEDDING_TOKEN_BUDGET", "19000")
os.environ.setdefault("EMBEDDING_MAX_CHARS", "80000")
os.environ.setdefault("EMBEDDING_CONCURRENCY", "8")
os.environ.setdefault("DATABASE_PATH", "data/processed/vectors_large.db")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.chunking.chunker import DocumentChunker  # noqa: E402
from src.config import settings  # noqa: E402
from src.embeddings.factory import build_embedder  # noqa: E402
from src.embeddings.google_embedding import GoogleEmbedding  # noqa: E402
from src.embeddings.local_embedding import LocalEmbedding  # noqa: E402
from src.ingestion.ingestion_service import compute_checksum  # noqa: E402
from src.loaders.document_loader import DocumentLoader  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.vectordb.schema import CHUNKS_TABLE, DOCUMENT_REGISTRY_TABLE  # noqa: E402
from src.vectordb.vector_store import VectorStore  # noqa: E402

logger = get_logger(__name__)

DATA_DIR = REPO_ROOT / "data" / "documents_large"
DB_PATH = REPO_ROOT / "data" / "processed" / "vectors_large.db"
WAVE_SIZE = 2048


def repair_partial_documents(vector_store: VectorStore) -> list[str]:
    """Find registered documents that don't have all their chunks stored
    (e.g. a crash between the wave holding a doc's head chunks and the
    one holding its tail), delete them and let the run re-ingest them.
    """
    stored_rows = vector_store.database.query(
        f"SELECT document_id, COUNT(*) AS n FROM {CHUNKS_TABLE} GROUP BY document_id"
    )
    stored = {row["document_id"]: int(row["n"]) for row in stored_rows}
    reg_rows = vector_store.database.query(
        f"SELECT document_id, chunk_count FROM {DOCUMENT_REGISTRY_TABLE}"
    )
    broken = []
    for row in reg_rows:
        doc_id = row["document_id"]
        if stored.get(doc_id, 0) != int(row["chunk_count"]):
            logger.warning(
                "Repairing document '%s': registered %s chunks, found %s",
                doc_id, row["chunk_count"], stored.get(doc_id, 0),
            )
            vector_store.delete_document(doc_id)
            broken.append(doc_id)
    return broken

_START = time.perf_counter()
_GRAND_TOTAL = 0
_WAVE_OFFSET = 0


def _print_progress(done: int, total: int, elapsed: float, rate: float, eta: float):
    print(
        f"PROGRESS embedded={done}/{total} "
        f"pct={100.0 * done / total:.1f}% rate={rate:.0f}/s "
        f"elapsed={int(elapsed)}s eta={int(eta)}s",
        flush=True,
    )


def _overall_progress_cb(done: int, total: int, elapsed: float, rate: float, eta: float):
    """Convert per-wave progress into run-wide progress.

    Rate/ETA are recomputed against the true run start so they stay
    meaningful across waves; wave-local counters are ignored.
    """
    overall = _WAVE_OFFSET + done
    elapsed_total = max(time.perf_counter() - _START, 1e-6)
    run_rate = overall / elapsed_total
    run_eta = (_GRAND_TOTAL - overall) / run_rate if run_rate > 0 else 0
    _print_progress(overall, _GRAND_TOTAL, elapsed_total, run_rate, run_eta)


class ProgressLocalEmbedding(LocalEmbedding):
    """LocalEmbedding that reports batch progress while embedding."""

    progress_cb = None

    def _ensure_model(self):
        # Use every logical core: measured 3.6 vs 2.5 chunks/s on 2 KB
        # legal chunks with onnxruntime default thread settings.
        import os

        if self._model is None:
            from fastembed import TextEmbedding

            logger.info(
                "Loading local embedding model %s (threads=%s)...",
                self.model,
                os.cpu_count(),
            )
            self._model = TextEmbedding(
                model_name=self.model,
                batch_size=self.batch_size,
                threads=os.cpu_count(),
            )
        return self._model

    def embed_documents(self, chunks):
        logger = get_logger(__name__)
        if not chunks:
            return []

        logger.info(
            "Embedding %d chunks using local %s (batch_size=%d)",
            len(chunks), self.model, self.batch_size,
        )
        model = self._ensure_model()

        total = len(chunks)
        done = 0
        t0 = time.perf_counter()
        embedded = []

        for start in range(0, total, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            texts = [chunk["text"] for chunk in batch]
            vectors = list(model.embed(texts))

            if vectors and self._dimensions is None:
                self._dimensions = len(vectors[0])

            for chunk, vector in zip(batch, vectors):
                embedded.append(
                    {
                        "text": chunk["text"],
                        "embedding": [float(v) for v in vector],
                        "metadata": {
                            "chunk_id": chunk["chunk_id"],
                            "document_id": chunk["document_id"],
                            **chunk["metadata"],
                        },
                    }
                )

            done += len(batch)
            elapsed = max(time.perf_counter() - t0, 1e-6)
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            cb = self.progress_cb or _print_progress
            cb(done, total, elapsed, rate, eta)

        logger.info(
            "Generated %d embeddings (dimensions=%s)", len(embedded), self._dimensions
        )
        return embedded


def build_progress_embedder():
    """Wrap the configured backend so every batch reports overall progress."""
    backend = settings.embedding_backend
    if backend == "google":
        return GoogleEmbedding(
            project_id=settings.gcp_project_id,
            location=settings.gcp_location,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_chars=settings.embedding_max_chars,
            max_request_tokens=settings.embedding_token_budget,
            concurrency=settings.embedding_concurrency,
            progress_cb=_overall_progress_cb,
        )
    if backend == "local":
        local_embedder = ProgressLocalEmbedding(
            model=settings.local_embedding_model,
            batch_size=settings.embedding_batch_size,
        )
        local_embedder.progress_cb = _overall_progress_cb
        return local_embedder
    return build_embedder(settings)


def main() -> None:
    global _GRAND_TOTAL, _WAVE_OFFSET

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true",
                        help="delete the target database first (default: resume)")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="ingest only the first N files (smoke test)")
    args = parser.parse_args()

    print(
        f"Ingesting {DATA_DIR} -> {DB_PATH} "
        f"(backend={settings.embedding_backend}, "
        f"model={settings.embedding_model if settings.embedding_backend == 'google' else settings.local_embedding_model}, "
        f"batch={settings.embedding_batch_size}, token_budget={settings.embedding_token_budget}, "
        f"concurrency={settings.embedding_concurrency}, waves={WAVE_SIZE})",
        flush=True,
    )

    if args.fresh:
        for suffix in ("", ".wal", ".shm"):
            p = Path(str(DB_PATH) + suffix)
            if p.exists():
                p.unlink()
                print(f"Removed {p.name}", flush=True)

    vector_store = VectorStore(database_path=str(DB_PATH), default_top_k=settings.top_k)
    vectors_before = vector_store.count()
    repaired = repair_partial_documents(vector_store)
    if repaired:
        print(f"Repaired (re-queued) {len(repaired)} partially-stored document(s)", flush=True)

    # --- Load ---
    loader = DocumentLoader()
    documents = loader.load_directory(DATA_DIR)
    if args.limit:
        documents = documents[: args.limit]
    print(f"Loaded {len(documents)} documents", flush=True)
    if not documents:
        print("Nothing to ingest.", flush=True)
        return

    # --- Deduplicate against the registry (resume support) ---
    new_documents = []
    skipped = 0
    for document in documents:
        source = document.get("metadata", {}).get("source")
        if not source:
            new_documents.append(document)
            continue
        checksum = compute_checksum(Path(source))
        existing = vector_store.is_document_ingested(checksum)
        if existing:
            skipped += 1
            continue
        document["id"] = checksum
        document["metadata"]["checksum"] = checksum
        new_documents.append(document)
    print(
        f"New documents: {len(new_documents)} "
        f"(already ingested, skipped: {skipped})",
        flush=True,
    )
    if not new_documents:
        print("All documents already ingested; DONE", flush=True)
        return

    # --- Chunk once (local CPU, no cost) ---
    chunker = DocumentChunker(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = chunker.chunk_documents(new_documents)
    _GRAND_TOTAL = len(chunks)
    print(f"Chunks created: {_GRAND_TOTAL}", flush=True)
    if not chunks:
        print("No chunks produced; DONE", flush=True)
        return

    doc_counts = Counter(chunk["document_id"] for chunk in chunks)
    doc_meta = {
        doc["id"]: (
            doc["metadata"].get("checksum", doc["id"]),
            doc["filename"],
            doc["metadata"].get("source"),
        )
        for doc in new_documents
    }

    embedder = build_progress_embedder()
    waves = [chunks[i : i + WAVE_SIZE] for i in range(0, len(chunks), WAVE_SIZE)]
    print(f"Processing {_GRAND_TOTAL} chunks in {len(waves)} wave(s) of <= {WAVE_SIZE}", flush=True)

    embedded_so_far = 0
    errors: list[str] = []
    doc_inserted = Counter()

    # --- Embed + insert + register, wave by wave ---
    for i, wave in enumerate(waves, start=1):
        _WAVE_OFFSET = embedded_so_far
        try:
            embedded = embedder.embed_documents(wave)
        except Exception as exc:  # noqa: BLE001
            print(
                f"\nWAVE {i}/{len(waves)} FAILED: {exc}\n"
                f"Nothing after wave {i - 1} was lost. "
                f"Re-run this command to resume.",
                flush=True,
            )
            raise SystemExit(1)

        inserted = vector_store.insert_many(embedded)
        embedded_so_far += len(embedded)

        # Register a document only once EVERY chunk of it is stored -
        # documents can straddle wave boundaries, and registering early
        # would make a resume skip their tail chunks.
        doc_inserted.update(chunk["metadata"]["document_id"] for chunk in embedded)
        newly_complete = [
            doc_id for doc_id in {chunk["metadata"]["document_id"] for chunk in embedded}
            if doc_inserted[doc_id] >= doc_counts[doc_id]
        ]
        for doc_id in newly_complete:
            checksum, filename, source = doc_meta[doc_id]
            vector_store.register_document(
                document_id=doc_id,
                checksum=checksum,
                filename=filename,
                source=source,
                chunk_count=doc_counts[doc_id],
            )
        skipped_by_api = getattr(embedder, "skipped_chunks", 0)
        print(
            f"WAVE {i}/{len(waves)}: embedded={len(embedded)} inserted={inserted} "
            f"registered={len(newly_complete)} docs "
            f"run_total={embedded_so_far}/{_GRAND_TOTAL}"
            + (f" api_skipped={skipped_by_api}" if skipped_by_api else ""),
            flush=True,
        )

    # --- Summary ---
    total_time = time.perf_counter() - _START
    total_vectors = vector_store.count()
    print("\n================ INGESTION COMPLETE ================", flush=True)
    print(f"Documents loaded      : {len(documents)}", flush=True)
    print(f"Already ingested      : {skipped}", flush=True)
    print(f"New documents         : {len(new_documents)}", flush=True)
    print(f"Chunks created        : {_GRAND_TOTAL}", flush=True)
    print(f"Vectors stored        : {embedded_so_far}", flush=True)
    print(f"Total vectors in store: {total_vectors}", flush=True)
    db_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"DB size               : {db_mb:.1f} MB", flush=True)
    print(f"Total time            : {int(total_time)}s ({total_time/60:.1f} min)", flush=True)
    stored_delta = total_vectors - vectors_before
    if embedded_so_far != _GRAND_TOTAL or stored_delta != _GRAND_TOTAL:
        print(
            "WARNING: stored count != chunk count - inspect logs before benchmarking",
            flush=True,
        )
    print("DONE", flush=True)

    vector_store.close()


if __name__ == "__main__":
    main()