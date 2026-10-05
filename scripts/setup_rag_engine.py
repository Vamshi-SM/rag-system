#!/usr/bin/env python3
"""Create the Vertex AI RAG Engine corpus for the latency benchmark.

Uses the modern ``agentplatform`` client (the vertexai.preview.rag shim
is deprecated and cannot express serverless mode).

Stages (all idempotent, run any subset):

    --create   switch the RAG Engine to SERVERLESS mode (the default
               Spanner Basic mode bills ~$70-130/month; serverless is
               pay-per-use) + create the corpus (default embedding
               model: text-embedding-004, 768-dim - same as the local
               benchmark DB) + persist RAG_CORPUS_NAME to .env
    --upload   upload data/documents_large to the staging GCS bucket
               (skips files already present)
    --import   import files into the corpus with token chunking
               (chunk_size=700 / overlap=150, matching the local
               chunker) - long-running LRO, safe to re-run
    --status   corpus info + file count

Default with no flags: create + upload + import + status.

Requires ADC for a project with billing + Vertex AI enabled:
    gcloud auth application-default login
    gcloud auth application-default set-quota-project <project>
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import warnings

warnings.filterwarnings("ignore")

import agentplatform
from agentplatform._genai.types.common import (
    RagCorpus,
    RagEmbeddingModelConfig,
    RagEngineConfig,
    RagFileChunkingConfig,
    RagFileTransformationConfig,
    RagManagedDbConfig,
    RagManagedDbConfigServerless,
    RagVectorDbConfig,
    RagVectorDbConfigRagManagedDb,
    RagEmbeddingModelConfigVertexPredictionEndpoint,
    ImportRagFilesConfig,
)
from google.cloud import storage
from google.genai import types as gtypes

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = REPO_ROOT / "data" / "documents_large"
ENV_PATH = REPO_ROOT / ".env"

ALLOWED_EXTS = {".md", ".txt", ".pdf", ".docx"}


def bucket_name() -> str:
    if settings.rag_gcs_bucket:
        return settings.rag_gcs_bucket
    # Region is part of the name so a region switch (e.g. serverless is
    # us-central1-only) creates a co-located bucket instead of reusing
    # one pinned to the old region.
    return f"{settings.gcp_project_id}-rag-benchmark-{settings.gcp_location.replace('-', '')}"


def engine_name() -> str:
    return f"projects/{settings.gcp_project_id}/locations/{settings.gcp_location}/ragEngineConfig"


def ensure_serverless(client) -> bool:
    """Switch the RAG Engine to serverless mode. Returns True if a switch
    was performed. Only safe/possible while the engine has no corpora;
    this project starts fresh.
    """
    cfg = client.rag.get_config()
    current = cfg.rag_managed_db_config
    if current is not None and current.serverless is not None:
        print("Engine already in SERVERLESS mode", flush=True)
        return False
    print(f"Engine mode before: {current}", flush=True)
    updated = client.rag.update_config(
        updated_config=RagEngineConfig(
            name=engine_name(),
            rag_managed_db_config=RagManagedDbConfig(
                serverless=RagManagedDbConfigServerless()
            ),
        )
    )
    mode = updated.rag_managed_db_config
    print(f"Engine mode after : {mode}", flush=True)
    if mode is None or mode.serverless is None:
        raise SystemExit("update_config did not result in serverless mode - aborting")
    return True


def ensure_bucket(client: storage.Client):
    name = bucket_name()
    try:
        bucket = client.get_bucket(name)
        print(f"Bucket exists: gs://{name} (location={bucket.location})", flush=True)
        return bucket
    except Exception:  # noqa: BLE001 - not found -> create
        bucket = client.bucket(name)
        bucket.storage_class = "STANDARD"
        bucket = client.create_bucket(bucket, location=settings.gcp_location)
        print(f"Created bucket: gs://{name} (location={bucket.location})", flush=True)
        return bucket


def stage_upload(bucket) -> int:
    """Upload data/documents_large, skipping files already in the bucket."""
    files = sorted(
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    )
    print(
        f"Found {len(files)} files to stage "
        f"({sum(f.stat().st_size for f in files)/1e6:.1f} MB)",
        flush=True,
    )

    existing = {blob.name for blob in bucket.list_blobs()}

    def upload(path: Path) -> str | None:
        blob_name = f"documents_large/{path.name}"
        if blob_name in existing:
            return None
        bucket.blob(blob_name).upload_from_filename(str(path))
        return blob_name

    uploaded = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(upload, f) for f in files]
        for fut in as_completed(futures):
            if fut.result():
                uploaded += 1
    print(f"Uploaded {uploaded} new files -> gs://{bucket.name}/documents_large/", flush=True)
    return uploaded


def find_corpus(client) -> RagCorpus | None:
    display = settings.rag_corpus_display_name
    resp = client.rag.list_corpora()
    for corpus in resp.rag_corpora or []:
        if corpus.display_name == display:
            return corpus
    return None


def create_corpus(client) -> RagCorpus:
    # In serverless mode the engine manages storage; specifying
    # rag_managed_db (or any vector db) is rejected with INVALID_ARGUMENT.
    # Pin text-embedding-004 explicitly: the serverless default proved to
    # be text-embedding-005, and the local benchmark DB uses 004 (768-dim)
    # - the comparison requires embedding parity.
    corpus = client.rag.create_corpus(
        rag_corpus=RagCorpus(
            display_name=settings.rag_corpus_display_name,
            description="Latency benchmark: 100 MB legal corpus",
            rag_vector_db_config=RagVectorDbConfig(
                rag_embedding_model_config=RagEmbeddingModelConfig(
                    vertex_prediction_endpoint=RagEmbeddingModelConfigVertexPredictionEndpoint(
                        endpoint=(
                            f"projects/{settings.gcp_project_id}/locations/"
                            f"{settings.gcp_location}/publishers/google/models/"
                            f"{settings.embedding_model}"
                        )
                    )
                )
            ),
        )
    )
    print(f"Created corpus: {corpus.name}", flush=True)
    if corpus.rag_embedding_model_config is not None:
        print(f"  embedding: {corpus.rag_embedding_model_config}", flush=True)
    print(f"  vector db: {corpus.rag_vector_db_config}", flush=True)
    return corpus


def recreate_corpus(client) -> RagCorpus:
    """Delete the existing corpus and create it with pinned settings.

    Preferred over per-file repair when most files errored: corpus
    delete/create are single control-plane calls, while per-file deletes
    are quota-paced (~5/min observed on fresh projects).
    """
    existing = find_corpus(client)
    if existing is None:
        print("No existing corpus to recreate", flush=True)
    else:
        _with_quota_retry(client.rag.delete_corpus, name=existing.name)
        print(f"Deleted corpus: {existing.name}", flush=True)
        time.sleep(10)
    return create_corpus(client)


def _with_quota_retry(fn, *args, **kwargs):
    """Control-plane (and import) calls are quota-gated at a few requests
    per minute on fresh projects; back off past the minute boundary."""
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc).upper():
                wait = 65 * (attempt + 1)
                print(f"    quota hit; backing off {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise SystemExit("Quota retries exhausted")


def full_snapshot(client, corpus_name: str) -> dict[str, tuple[str, str]]:
    """All files -> {display_name: (state, resource_name)} across every page."""
    out: dict[str, tuple[str, str]] = {}
    token = None
    while True:
        cfg = {"page_size": 100}
        if token:
            cfg["page_token"] = token
        resp = _with_quota_retry(client.rag.list_files, name=corpus_name, config=cfg)
        for f in resp.rag_files or []:
            state = str(getattr(getattr(f, "file_status", None), "state", None) or "UNKNOWN")
            out[f.display_name] = (state.replace("RagFileState.", ""), f.name)
        token = getattr(resp, "next_page_token", None)
        if not token:
            break
    return out


def files_snapshot(client, corpus_name: str) -> tuple[int, dict]:
    """(total_files, state_histogram) across ALL pages."""
    try:
        snap = full_snapshot(client, corpus_name)
    except Exception as exc:  # noqa: BLE001
        print(f"list_files failed: {str(exc)[:200]}", flush=True)
        return 0, {}
    states: dict = {}
    for _name, (state, _n) in snap.items():
        states[state] = states.get(state, 0) + 1
    return len(snap), states


def sync_corpus(client, corpus_name: str, rpm: int = 500) -> bool:
    """Bring the corpus to exactly the staged file set. Returns True when
    every staged file is ACTIVE.

    Duplicate-safe rule: a file present in ANY state is never re-imported
    (a killed run leaves files in UNKNOWN state; re-importing them would
    double-embed AND double-index them). ERROR files are deleted first,
    which is what makes their re-import possible at all - the API skips
    files whose URI is already registered, regardless of state.
    """
    staged = sorted(
        p.name for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    )
    snap = full_snapshot(client, corpus_name)
    active = sum(1 for n, (s, _r) in snap.items() if s == "ACTIVE")
    errored = {n: r for n, (s, r) in snap.items() if s == "ERROR"}
    print(
        f"Corpus: {len(snap)} files registered ({active} ACTIVE, "
        f"{len(errored)} ERROR); staged: {len(staged)}",
        flush=True,
    )
    if errored:
        print(f"Deleting {len(errored)} ERROR files (quota-paced): "
              f"{sorted(errored)[:5]}", flush=True)
        for n, r in errored.items():
            _with_quota_retry(client.rag.delete_file, name=r)
        snap = full_snapshot(client, corpus_name)
    todo = [n for n in staged if n not in snap]
    print(f"Files to import: {len(todo)}", flush=True)
    if not todo:
        print("Corpus matches staged files. Nothing to do.", flush=True)
        return True
    return import_batched(client, corpus_name, todo, rpm=rpm)


def repair_import(client, corpus_name: str) -> None:
    """Alias kept for continuity - same logic as import_files."""
    run_until_complete(client, corpus_name)


def import_batched(client, corpus_name: str, file_names: list[str], rpm: int = 500) -> bool:
    """Import files in small batches at a reduced embedding rate.

    The serverless 'Rag Managed Vector Search 2.0' insert path 429s under
    sustained load (observed with a single 223-file import at 2000 rpm:
    44 files imported, 179 failed). 20 files/batch @ 500 rpm stays under
    the quota; every batch carries a failure sink so errors are readable.

    Returns True when every requested file ended up ACTIVE; False if a
    batch left ERROR files (caller cools down, deletes and retries).
    """
    sink_prefix = "failure_sink/"
    gcs = storage.Client(project=settings.gcp_project_id)
    bucket = gcs.bucket(bucket_name())

    batch_size = 20
    rpm = 500
    n_batches = (len(file_names) + batch_size - 1) // batch_size
    for i in range(0, len(file_names), batch_size):
        batch = file_names[i : i + batch_size]
        # The import API rejects a sink path that already exists, so the
        # prefix must be emptied before EVERY batch.
        for b in bucket.list_blobs(prefix=sink_prefix):
            b.delete()
        uris = [f"gs://{bucket_name()}/documents_large/{n}" for n in batch]
        print(f"Import batch {i//batch_size + 1}/{n_batches}: "
              f"{len(batch)} files @ {rpm} rpm...", flush=True)
        t0 = time.perf_counter()
        _with_quota_retry(
            client.rag.import_files,
            name=corpus_name,
            import_config=ImportRagFilesConfig(
                gcs_source=gtypes.GcsSource(uris=uris),
                rag_file_transformation_config=RagFileTransformationConfig(
                    rag_file_chunking_config=RagFileChunkingConfig(
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                    )
                ),
                partial_failure_gcs_sink=gtypes.GcsDestination(
                    output_uri_prefix=f"gs://{bucket_name()}/{sink_prefix}"
                ),
                max_embedding_requests_per_min=rpm,
            ),
        )
        snap = full_snapshot(client, corpus_name)
        states: dict = {}
        for _n, (s, _r) in snap.items():
            states[s] = states.get(s, 0) + 1
        print(f"  batch took {time.perf_counter()-t0:.0f}s; corpus states: {states}", flush=True)
        if states.get("ERROR"):
            print("Batch produced ERROR files - failure records:", flush=True)
            for b in bucket.list_blobs(prefix=sink_prefix):
                for line in b.download_as_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if rec.get("Status") != "OK":
                        print(f"  FAILED {rec.get('Filename', '?')}: "
                              f"{str(rec.get('Message', ''))[:160]}", flush=True)
            return False
    return True


def corpus_files_count(client, corpus_name: str) -> int:
    return files_snapshot(client, corpus_name)[0]


def run_until_complete(client, corpus_name: str, rounds: int = 6) -> None:
    """Sync with cooldown retries: each round deletes any ERROR files and
    imports the remainder; 429-damaged batches cost one re-embedded file
    set, not the whole run. Decays rpm a notch per round as a fallback.
    """
    rpm = 500
    for round_no in range(1, rounds + 1):
        print(f"--- sync round {round_no}/{rounds} (rpm={rpm}) ---", flush=True)
        if sync_corpus(client, corpus_name, rpm=rpm):
            print("Corpus fully imported.", flush=True)
            return
        time.sleep(90)
        rpm = max(250, rpm - 50)
    raise SystemExit("Import still incomplete after retry rounds - run --status")


def import_files(client, corpus_name: str) -> None:
    """Plain import - same duplicate-safe sync as --repair."""
    run_until_complete(client, corpus_name)


def write_env(corpus_name: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    out = []
    seen = False
    for line in lines:
        if line.startswith("RAG_CORPUS_NAME="):
            out.append(f"RAG_CORPUS_NAME={corpus_name}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"RAG_CORPUS_NAME={corpus_name}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote RAG_CORPUS_NAME to {ENV_PATH.name}", flush=True)


def show_status(client, corpus: RagCorpus) -> None:
    print(f"Corpus: {corpus.name}", flush=True)
    print(f"  state   : {corpus.corpus_status}", flush=True)
    print(f"  files   : {corpus.rag_files_count} (list_files: "
          f"{corpus_files_count(client, corpus.name)})", flush=True)
    print(f"  embedding: {corpus.rag_embedding_model_config}", flush=True)
    print(f"  vector db: {corpus.rag_vector_db_config}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--recreate", action="store_true",
                        help="delete the existing corpus and recreate it")
    parser.add_argument("--repair", action="store_true",
                        help="delete ERROR files and import the rest in throttled batches")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--import", dest="do_import", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    do_all = not any(
        [args.create, args.recreate, args.repair, args.upload, args.do_import, args.status]
    )

    if not settings.gcp_project_id:
        raise SystemExit("Set GCP_PROJECT_ID in .env first")
    print(f"Project={settings.gcp_project_id} Location={settings.gcp_location}", flush=True)

    client = agentplatform.Client(
        project=settings.gcp_project_id, location=settings.gcp_location
    )

    corpus = None
    if args.repair:
        corpus = find_corpus(client)
        if corpus is None:
            raise SystemExit("No corpus - run with --create first")
        repair_import(client, corpus.name)

    if args.create or args.recreate or do_all:
        if args.recreate:
            ensure_serverless(client)
            corpus = recreate_corpus(client)
        else:
            ensure_serverless(client)
            corpus = find_corpus(client)
            if corpus is None:
                corpus = create_corpus(client)
            else:
                print(f"Corpus exists: {corpus.name}", flush=True)
        write_env(corpus.name)

    if args.upload or do_all:
        gcs = storage.Client(project=settings.gcp_project_id)
        bucket = ensure_bucket(gcs)
        stage_upload(bucket)

    if args.do_import or do_all:
        if corpus is None:
            corpus = find_corpus(client)
            if corpus is None:
                raise SystemExit("No corpus - run with --create first")
        import_files(client, corpus.name)

    if args.status or do_all:
        if corpus is None:
            corpus = find_corpus(client)
        if corpus is None:
            print("No corpus found", flush=True)
        else:
            show_status(client, corpus)


if __name__ == "__main__":
    main()