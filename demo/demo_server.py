"""Side-by-side user-experience demo: Local hybrid RAG vs Vertex AI RAG Engine.

Run from the repo root:

    python demo\\demo_server.py

Serves demo/index.html at / and exposes two SSE endpoints that stream the
SAME event shape (event: sources -> event: token* -> event: done) so the
page can render both engines identically:

  POST /local/stream  proxies the REAL production API (uvicorn
                      src.api.main:app on port 8000, POST /query/stream),
                      so the left panel is the actual shipped path, not a
                      reimplementation.

  POST /rag/stream    calls Vertex AI RAG Engine grounding directly
                      (gemini-2.5-flash + VertexRagStore tool), streaming
                      tokens and extracting citation sources from the
                      grounding metadata as Google returns them.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

# Allow running as `python demo\demo_server.py` from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.config import settings

INDEX_HTML = REPO_ROOT / "demo" / "index.html"

LOCAL_API = "http://127.0.0.1:8000"
RAG_TOP_K = 5
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_UPLOAD_EXTS = {".pdf", ".docx", ".txt", ".md"}

# Gemini free-tier 429s clear in 30-60s; retry stream creation the same way
# the production pipeline does, but keep it short for a demo.
RAG_RETRY_DELAYS = (30.0, 60.0)

app = FastAPI(title="RAG face-off demo", docs_url=None, redoc_url=None)

_genai_client = None

# State of the latest RAG Engine import job (background thread updates this).
RAG_STATUS: dict = {"state": "idle", "filename": None, "detail": ""}

# Parsed chunks of recent uploads -> the instant-answer fast path: while
# Vertex is still importing, the RAG panel answers straight from the file's
# text in-context (the pattern a production deployment would use instead of
# making users wait for the corpus import).
UPLOADS: dict[str, list[dict]] = {}

# Progress of the running test suite (page polls /test-suite/status).
SUITE_STATUS: dict = {"stage": "idle", "detail": "", "done": 0, "total": 0}

FIXTURE_PDF = REPO_ROOT / "benchmarks" / "fixtures" / "kestrel_case_file.pdf"
FIXTURE_PREFIX = "kestrel_case_file"
QUESTIONS_JSONL = REPO_ROOT / "benchmarks" / "kestrel_case_questions.jsonl"
# older test fixtures also get purged by the suite
PURGE_PREFIXES = ("kestrel_case_file", "kestrel_msa")

INSTANT_PROMPT = (
    "You are a legal research assistant. Answer the question using ONLY the "
    "context below. If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "for", "in", "to", "what", "which",
    "who", "how", "does", "do", "and", "or", "per", "from", "on", "under",
    "by", "with", "that", "this", "it", "its", "as", "at", "be", "was",
}


def _pick_chunks(question: str, chunks: list[dict], n: int = 5) -> list[dict]:
    """Tiny keyword scorer - enough to find the right chunk in one doc."""
    import re

    terms = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in _STOPWORDS]
    scored = []
    for c in chunks:
        text = c["text"].lower()
        scored.append((sum(text.count(w) for w in terms), c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:n]]


def _genai():
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )
    return _genai_client


class QuestionRequest(BaseModel):
    question: str


def _bucket_name() -> str:
    if settings.rag_gcs_bucket:
        return settings.rag_gcs_bucket
    return (
        f"{settings.gcp_project_id}-rag-benchmark-"
        f"{settings.gcp_location.replace('-', '')}"
    )


def _sanitize_filename(name: str) -> str:
    from pathlib import PurePosixPath

    safe = PurePosixPath(name.replace("\\", "/")).name
    keep = [c if (c.isalnum() or c in "._-") else "_" for c in safe]
    safe = "".join(keep).strip("._") or "document"
    return safe


def _rag_import_job(gcs_uri: str, display: str, blob_name: str) -> None:
    """Import one uploaded file into the RAG Engine corpus (background)."""
    import warnings

    warnings.filterwarnings("ignore")
    import agentplatform
    from google.cloud import storage as gcs_storage
    from agentplatform._genai.types.common import (
        ImportRagFilesConfig,
        RagFileChunkingConfig,
        RagFileTransformationConfig,
    )
    from google.genai import types as gtypes

    def quota_retry(fn, **kw):
        for attempt in range(6):
            try:
                return fn(**kw)
            except Exception as exc:  # noqa: BLE001
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc).upper():
                    wait = 65 * (attempt + 1)
                    RAG_STATUS["detail"] = f"Vertex quota backoff {wait}s..."
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("quota retries exhausted")

    try:
        RAG_STATUS.update(state="importing", filename=display,
                          detail="sending import request to Vertex...")
        gcs = gcs_storage.Client(project=settings.gcp_project_id)
        bucket = gcs.bucket(_bucket_name())
        # The import API rejects an existing failure-sink path.
        for b in bucket.list_blobs(prefix="failure_sink/"):
            b.delete()
        client = agentplatform.Client(
            project=settings.gcp_project_id, location=settings.gcp_location
        )
        quota_retry(
            client.rag.import_files,
            name=settings.rag_corpus_name,
            import_config=ImportRagFilesConfig(
                gcs_source=gtypes.GcsSource(uris=[gcs_uri]),
                rag_file_transformation_config=RagFileTransformationConfig(
                    rag_file_chunking_config=RagFileChunkingConfig(
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                    )
                ),
                partial_failure_gcs_sink=gtypes.GcsDestination(
                    output_uri_prefix=f"gs://{_bucket_name()}/failure_sink/"
                ),
                max_embedding_requests_per_min=500,
            ),
        )
        # import_files blocks until the LRO completes; poll until the file
        # is ACTIVE (or ERROR) so the page can show the real state.
        t0 = time.time()
        while time.time() - t0 < 900:
            time.sleep(20)
            try:
                resp = client.rag.list_files(
                    name=settings.rag_corpus_name, config={"page_size": 100}
                )
                for f in resp.rag_files or []:
                    if f.display_name != display:
                        continue
                    st = str(
                        getattr(getattr(f, "file_status", None), "state", None) or ""
                    ).replace("RagFileState.", "")
                    if st == "ACTIVE":
                        RAG_STATUS.update(state="active", detail="indexed by RAG Engine")
                        return
                    if st == "ERROR":
                        RAG_STATUS.update(state="error", detail="import errored at Vertex")
                        return
            except Exception as exc:  # noqa: BLE001
                RAG_STATUS["detail"] = f"polling: {str(exc)[:120]}"
        RAG_STATUS.update(state="error", detail="timed out waiting for ACTIVE state")
    except Exception as exc:  # noqa: BLE001
        RAG_STATUS.update(state="error", detail=str(exc)[:300])


@app.post("/upload")
async def upload(request: Request, filename: str = "") -> dict:
    """Save an uploaded document, index it locally (production /ingest),
    and kick off a RAG Engine import in the background."""
    from fastapi.responses import JSONResponse

    safe = _sanitize_filename(filename or "")
    ext = Path(safe).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        return JSONResponse(
            status_code=400,
            content={"detail": f"unsupported type '{ext}' - allowed: "
                               + ", ".join(sorted(ALLOWED_UPLOAD_EXTS))},
        )
    body = await request.body()
    if not body:
        return JSONResponse(status_code=400, content={"detail": "empty upload"})
    if len(body) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={"detail": "file too large (50 MB cap)"})

    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = Path(safe).stem
    unique_name = f"{stem}_{ts}{ext}"
    folder = UPLOAD_DIR / f"{stem}_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    doc_path = folder / unique_name
    doc_path.write_bytes(body)

    # Parse + chunk now so the RAG panel can answer instantly while Vertex
    # imports (same loaders/chunker the production pipeline uses).
    parsed_chunks = 0
    try:
        from src.chunking.chunker import DocumentChunker
        from src.loaders.document_loader import DocumentLoader

        doc = DocumentLoader().load_file(doc_path)
        if doc:
            chunks = DocumentChunker(
                settings.chunk_size, settings.chunk_overlap
            ).chunk_document(doc)
            UPLOADS[unique_name] = [
                {"text": c["text"], "page": c["metadata"].get("page")} for c in chunks
            ]
            while len(UPLOADS) > 3:
                UPLOADS.pop(next(iter(UPLOADS)))
            parsed_chunks = len(UPLOADS[unique_name])
    except Exception:  # noqa: BLE001 - instant mode is best-effort
        parsed_chunks = 0

    # Frozen question chips for the test fixture (no AI generation):
    # uploads matching the fixture name serve the fixed question set.
    fixture_questions = []
    if safe.lower().startswith(FIXTURE_PREFIX):
        try:
            for line in QUESTIONS_JSONL.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    fixture_questions.append({
                        "id": rec["id"],
                        "question": rec["question"],
                        "category": rec["category"],
                    })
        except Exception:  # noqa: BLE001
            fixture_questions = []

    # --- Local side: real production ingest (folder scan -> chunks -> embeddings)
    local = {"state": "ok"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=15.0)) as client:
            resp = await client.post(
                f"{LOCAL_API}/ingest", json={"folder": str(doc_path.parent)}
            )
            if resp.status_code != 200:
                local = {"state": "error", "detail": f"ingest {resp.status_code}: {resp.text[:200]}"}
            else:
                r = resp.json()
                local = {
                    "state": "ok",
                    "chunks": r.get("chunks"),
                    "vectors": r.get("stored_vectors"),
                    "duplicates": r.get("duplicates_skipped"),
                    "seconds": r.get("processing_time"),
                }
    except Exception as exc:  # noqa: BLE001
        local = {"state": "error", "detail": str(exc)[:200]}

    # --- RAG Engine side: stage to GCS, import in a background thread
    rag = {"state": "skipped", "detail": "local ingest failed; fix that first"}
    if local.get("state") == "ok":
        try:
            from google.cloud import storage as gcs_storage

            blob_name = f"uploads/{unique_name}"
            gcs = gcs_storage.Client(project=settings.gcp_project_id)
            gcs.bucket(_bucket_name()).blob(blob_name).upload_from_string(
                body, content_type="application/pdf"
            )
            RAG_STATUS.update(state="importing", filename=unique_name, detail="uploaded to GCS")
            threading.Thread(
                target=_rag_import_job,
                args=(f"gs://{_bucket_name()}/{blob_name}", unique_name, blob_name),
                daemon=True,
            ).start()
            rag = {"state": "importing", "filename": unique_name,
                   "detail": "Vertex is parsing/chunking/embedding this file; usually 1-5 min",
                   "instant_chunks": parsed_chunks}
        except Exception as exc:  # noqa: BLE001
            rag = {"state": "error", "detail": str(exc)[:200]}

    return {"local": local, "rag": rag, "indexed_name": unique_name,
            "fixture_questions": fixture_questions}


_last_probe = {"t": 0.0}


def _probe_file_state(client, display: str) -> str | None:
    """Paginated list_files lookup for one file; returns its state."""
    token = None
    while True:
        cfg: dict = {"page_size": 100}
        if token:
            cfg["page_token"] = token
        resp = client.rag.list_files(name=settings.rag_corpus_name, config=cfg)
        for f in resp.rag_files or []:
            if f.display_name == display:
                return str(
                    getattr(getattr(f, "file_status", None), "state", None) or ""
                ).replace("RagFileState.", "")
        token = getattr(resp, "next_page_token", None)
        if not token:
            break
    return None


def _purge_local(prefix: str) -> int:
    """Delete all local-DB chunks whose filename starts with ``prefix``."""
    from src.vectordb.vector_store import VectorStore

    vs = VectorStore(database_path=str(settings.database_path), default_top_k=1)
    try:
        if vs.is_postgres:
            rows = vs.database.query(
                "SELECT DISTINCT document_id FROM chunks "
                "WHERE metadata->>'filename' LIKE %s",
                (prefix + "%",),
            )
        else:
            rows = vs.database.query(
                "SELECT DISTINCT document_id FROM chunks "
                "WHERE json_extract(metadata, '$.filename') LIKE ?",
                (prefix + "%",),
            )
        n = 0
        for r in rows:
            n += vs.delete_document(r["document_id"])
        return n
    finally:
        vs.close()


def _purge_vertex(prefix: str) -> int:
    """Delete corpus files whose display_name starts with ``prefix``."""
    import warnings

    warnings.filterwarnings("ignore")
    import agentplatform

    client = agentplatform.Client(
        project=settings.gcp_project_id, location=settings.gcp_location
    )
    names: list[str] = []
    token = None
    while True:
        cfg: dict = {"page_size": 100}
        if token:
            cfg["page_token"] = token
        resp = client.rag.list_files(name=settings.rag_corpus_name, config=cfg)
        for f in resp.rag_files or []:
            if (f.display_name or "").startswith(prefix):
                names.append(f.name)
        token = getattr(resp, "next_page_token", None)
        if not token:
            break
    n = 0
    for nm in names:
        for attempt in range(6):
            try:
                client.rag.delete_file(name=nm)
                n += 1
                break
            except Exception as exc:  # noqa: BLE001
                if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc).upper()) and attempt < 5:
                    time.sleep(65 * (attempt + 1))
                else:
                    raise
    return n


def _parse_sse_text(text: str) -> dict:
    evs: dict = {"tokens": [], "sources": [], "mode": None, "error": None}
    for block in text.split("\n\n"):
        ev = data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[7:].strip()
            elif line.startswith("data: "):
                data = line[6:]
        if ev is None or data is None:
            continue
        try:
            payload = json.loads(data)
        except Exception:  # noqa: BLE001
            continue
        if ev == "token":
            evs["tokens"].append(payload)
        elif ev == "sources":
            evs["sources"] = payload.get("sources", [])
        elif ev == "mode":
            evs["mode"] = payload.get("mode")
        elif ev == "error":
            evs["error"] = payload.get("detail")
    evs["answer"] = "".join(evs["tokens"])
    return evs


def _call_stream(hc: httpx.Client, url: str, question: str) -> dict:
    """Buffered SSE call with TTFB (first token event) + total timing."""
    t0 = time.perf_counter()
    ttfb_ms = None
    parts: list[str] = []
    with hc.stream("POST", url, json={"question": question}) as resp:
        if resp.status_code != 200:
            return {"text": "", "ttfb_ms": None, "total_ms": None}
        for chunk in resp.iter_bytes():
            parts.append(chunk.decode("utf-8", "replace"))
            if ttfb_ms is None and "event: token" in parts[-1]:
                ttfb_ms = round((time.perf_counter() - t0) * 1000)
    total_ms = round((time.perf_counter() - t0) * 1000)
    evs = _parse_sse_text("".join(parts))
    evs["ttfb_ms"] = ttfb_ms
    evs["total_ms"] = total_ms
    return evs


def _judge(qrec: dict, answer: str, sources: list[dict], fixture_text: str | None) -> dict:
    """Verdicts incl. page-citation assertion, extraction recall, and a
    demo-grade groundedness score (% of answer sentences sharing a content
    word with the fixture text)."""
    import re

    a = answer.lower()
    src_names = [(s.get("filename") or "") for s in sources]
    # Records may omit the prefix; fall back to the current fixture.
    pref = (qrec.get("expected_filename_prefix") or FIXTURE_PREFIX).lower()
    src_ok = bool(pref) and any(pref in n.lower() for n in src_names)

    def _groundedness() -> float | None:
        if not fixture_text:
            return None
        stop = {
            "the", "and", "for", "that", "with", "this", "from", "under",
            "which", "were", "have", "has", "per", "case", "file",
        }
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        if not sents:
            return None
        grounded = 0
        for s in sents:
            ws = [
                w for w in re.findall(r"[a-z0-9]{4,}", s.lower())
                if w not in stop
            ]
            if any(w in fixture_text for w in ws):
                grounded += 1
        return round(grounded / len(sents), 2)

    if qrec.get("type") == "negative":
        refusal = any(
            m in a
            for m in [
                "not in the provided context", "not in the context", "not provided",
                "not specified", "not available", "don't know", "do not know",
                "not mentioned", "not contained", "no information", "does not contain",
                "do not contain", "cannot find", "could not find", "no mention",
                "not covered", "does not mention", "do not mention", "do not have", "not include",
                "does not include", "not stated", "no such", "cannot be found",
                "no verdict", "has not been", "pending",
            ]
        )
        return {"answer_ok": refusal, "source_ok": None, "page_ok": None,
                "groundedness": None, "recall": None}

    if qrec.get("type") == "extraction":
        amounts = qrec.get("expected_amounts", [])
        found = [amt for amt in amounts if amt in a]
        recall = round(len(found) / len(amounts), 2) if amounts else None
        return {"answer_ok": (recall or 0) >= 0.7, "source_ok": src_ok,
                "page_ok": None, "groundedness": _groundedness(),
                "recall": recall, "found_count": len(found)}

    ok = all(k.lower() in a for k in qrec.get("must_include", []))
    if qrec.get("must_include_any"):
        ok = ok and any(k.lower() in a for k in qrec["must_include_any"])
    page_ok = None
    exp_page = qrec.get("expected_page")
    if exp_page is not None:
        page_ok = any(
            pref in (s.get("filename") or "").lower() and s.get("page") == exp_page
            for s in sources
        )
    return {"answer_ok": ok, "source_ok": src_ok, "page_ok": page_ok,
            "groundedness": _groundedness(), "recall": None}


def _run_suite() -> dict:
    SUITE_STATUS.update(stage="purging", detail="removing previous fixture copies", done=0, total=0)
    local_purged = 0
    vertex_purged = 0
    for prefix in PURGE_PREFIXES:
        try:
            local_purged += _purge_local(prefix)
        except Exception as exc:  # noqa: BLE001
            SUITE_STATUS["detail"] = f"local purge: {str(exc)[:120]}"
        try:
            vertex_purged += _purge_vertex(prefix)
        except Exception as exc:  # noqa: BLE001
            SUITE_STATUS["detail"] = f"vertex purge: {str(exc)[:120]}"
    try:
        from google.cloud import storage as gcs_storage

        gcs = gcs_storage.Client(project=settings.gcp_project_id)
        bucket = gcs.bucket(_bucket_name())
        for prefix in PURGE_PREFIXES:
            for b in bucket.list_blobs(prefix=f"uploads/{prefix}"):
                b.delete()
        for b in bucket.list_blobs(prefix="failure_sink/"):
            b.delete()
    except Exception:  # noqa: BLE001
        pass

    # --- Stage fixture copy + ingest locally (production /ingest)
    SUITE_STATUS.update(stage="ingesting", detail="fixture -> local DB")
    ts = time.strftime("%Y%m%d_%H%M%S")
    folder = UPLOAD_DIR / f"kestrel_fixture_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    doc_path = folder / f"{FIXTURE_PREFIX}.pdf"
    doc_path.write_bytes(FIXTURE_PDF.read_bytes())
    ingest = {"state": "ok"}
    with httpx.Client(timeout=httpx.Timeout(None, connect=15.0)) as hc:
        resp = hc.post(f"{LOCAL_API}/ingest", json={"folder": str(folder)})
        if resp.status_code != 200:
            ingest = {"state": "error", "detail": f"ingest {resp.status_code}: {resp.text[:150]}"}

    # --- GCS stage + Vertex import (foreground wait: the suite needs the
    # file ACTIVE so the RAG side is measured in grounded mode)
    display = f"{FIXTURE_PREFIX}_{ts}.pdf"
    blob_name = f"uploads/{display}"
    if ingest.get("state") == "ok":
        SUITE_STATUS.update(stage="importing", detail="fixture -> Vertex corpus")
        try:
            from google.cloud import storage as gcs_storage

            gcs = gcs_storage.Client(project=settings.gcp_project_id)
            gcs.bucket(_bucket_name()).blob(blob_name).upload_from_string(
                FIXTURE_PDF.read_bytes(), content_type="application/pdf"
            )
            import warnings

            warnings.filterwarnings("ignore")
            import agentplatform
            from agentplatform._genai.types.common import (
                ImportRagFilesConfig,
                RagFileChunkingConfig,
                RagFileTransformationConfig,
            )
            from google.genai import types as gtypes

            client = agentplatform.Client(
                project=settings.gcp_project_id, location=settings.gcp_location
            )
            for attempt in range(6):
                try:
                    client.rag.import_files(
                        name=settings.rag_corpus_name,
                        import_config=ImportRagFilesConfig(
                            gcs_source=gtypes.GcsSource(
                                uris=[f"gs://{_bucket_name()}/{blob_name}"]
                            ),
                            rag_file_transformation_config=RagFileTransformationConfig(
                                rag_file_chunking_config=RagFileChunkingConfig(
                                    chunk_size=settings.chunk_size,
                                    chunk_overlap=settings.chunk_overlap,
                                )
                            ),
                            partial_failure_gcs_sink=gtypes.GcsDestination(
                                output_uri_prefix=f"gs://{_bucket_name()}/failure_sink/"
                            ),
                            max_embedding_requests_per_min=500,
                        ),
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc).upper()) and attempt < 5:
                        SUITE_STATUS["detail"] = f"quota backoff {65 * (attempt + 1)}s"
                        time.sleep(65 * (attempt + 1))
                    else:
                        raise
            t0 = time.time()
            state = None
            while time.time() - t0 < 600:
                time.sleep(20)
                state = _probe_file_state(client, display)
                if state in ("ACTIVE", "ERROR"):
                    break
            RAG_STATUS.update(
                state="active" if state == "ACTIVE" else "error",
                filename=display,
                detail="indexed by RAG Engine" if state == "ACTIVE" else f"import state: {state}",
            )
        except Exception as exc:  # noqa: BLE001
            RAG_STATUS.update(state="error", filename=display, detail=str(exc)[:200])
    else:
        display = f"{FIXTURE_PREFIX}_{ts}.pdf"

    # --- Run the questions through both engines
    questions = [
        json.loads(line)
        for line in QUESTIONS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    SUITE_STATUS.update(stage="asking", detail="", done=0, total=len(questions))
    results = []
    fixture_text = None
    fname = RAG_STATUS.get("filename")
    if fname and UPLOADS.get(fname):
        fixture_text = "".join(c["text"].lower() for c in UPLOADS[fname])
    with httpx.Client(timeout=httpx.Timeout(None, connect=15.0)) as hc:
        for i, qrec in enumerate(questions, start=1):
            SUITE_STATUS.update(stage="asking", done=i - 1,
                                detail=f"Q{i} [{qrec['category']}]: {qrec['question'][:60]}")
            loc = _call_stream(hc, "http://127.0.0.1:8001/local/stream", qrec["question"])
            rag = _call_stream(hc, "http://127.0.0.1:8001/rag/stream", qrec["question"])
            lr = _judge(qrec, loc["answer"], loc["sources"], fixture_text)
            rr = _judge(qrec, rag["answer"], rag["sources"], fixture_text)
            results.append({
                "id": qrec["id"],
                "question": qrec["question"],
                "expected_answer": qrec["expected_answer"],
                "type": qrec["type"],
                "category": qrec["category"],
                "local": {"answer": loc["answer"], "mode": loc["mode"],
                          "sources": [s.get("filename") for s in loc["sources"]],
                          "error": loc["error"], "ttfb_ms": loc["ttfb_ms"],
                          "total_ms": loc["total_ms"], **lr},
                "rag": {"answer": rag["answer"], "mode": rag["mode"],
                        "sources": [s.get("filename") for s in rag["sources"]],
                        "error": rag["error"], "ttfb_ms": rag["ttfb_ms"],
                        "total_ms": rag["total_ms"], **rr},
            })

    def _side_pass(rows, side):
        return sum(
            1 for r in rows
            if r[side].get("answer_ok") and not r[side].get("error")
            and (r[side].get("source_ok") in (True, None))
        )

    summary = {"local_pass": _side_pass(results, "local"),
               "rag_pass": _side_pass(results, "rag"),
               "total": len(results),
               "purged": {"local_chunks": local_purged, "vertex_files": vertex_purged},
               "fixture": display,
               "rag_state": RAG_STATUS["state"]}
    SUITE_STATUS.update(stage="done", detail="complete", done=len(results), total=len(results))
    return {"summary": summary, "results": results}


@app.post("/test-suite")
def test_suite() -> dict:
    try:
        return _run_suite()
    except Exception as exc:  # noqa: BLE001
        SUITE_STATUS.update(stage="error", detail=str(exc)[:300])
        return {"error": str(exc)[:300]}


# Cached suggested questions per uploaded document (generated lazily).
SUGGESTIONS: dict[str, list[str]] = {}


@app.get("/suggestions")
def suggestions(name: str = "") -> dict:
    """Suggested questions for an uploaded document, generated from its
    parsed text by Gemini (cached per document)."""
    chunks = UPLOADS.get(name)
    if not chunks:
        return {"suggestions": [], "note": "unknown or expired document"}
    if name in SUGGESTIONS:
        return {"suggestions": SUGGESTIONS[name]}
    sample = "\n\n".join(c["text"][:1600] for c in chunks[:3])
    prompt = (
        "You help users explore a document. Based ONLY on the document text "
        "below, write 5 short, specific questions a user might ask about it. "
        "One question per line. No numbering, no quotes, no extra words.\n\n"
        f"Document: {name}\n\n{sample}"
    )
    try:
        resp = _genai().models.generate_content(
            model=settings.gemini_model, contents=prompt
        )
        qs = [
            ln.strip().lstrip("-•0123456789. ")
            for ln in (resp.text or "").splitlines()
            if ln.strip()
        ]
        qs = [q for q in qs if 8 < len(q) < 140][:5]
        SUGGESTIONS[name] = qs
        return {"suggestions": qs}
    except Exception as exc:  # noqa: BLE001
        return {"suggestions": [], "note": str(exc)[:120]}


@app.get("/test-suite/status")
def suite_status() -> dict:
    return dict(SUITE_STATUS)


_last_probe = {"t": 0.0}


@app.get("/rag-status")
def rag_status(name: str = "") -> dict:
    """Authoritative status: the import thread can hang inside the SDK's
    LRO wait while Vertex itself finishes, so trust list_files, not the
    thread. Live-probes the corpus (throttled to one probe / 25s)."""
    if RAG_STATUS["state"] == "active":
        return RAG_STATUS
    if name and time.time() - _last_probe["t"] > 25:
        _last_probe["t"] = time.time()
        try:
            import warnings

            warnings.filterwarnings("ignore")
            import agentplatform

            client = agentplatform.Client(
                project=settings.gcp_project_id, location=settings.gcp_location
            )
            st = _probe_file_state(client, name)
            if st == "ACTIVE":
                RAG_STATUS.update(state="active", detail="indexed by RAG Engine")
            elif st == "ERROR":
                RAG_STATUS.update(state="error", detail="import errored at Vertex")
        except Exception as exc:  # noqa: BLE001
            RAG_STATUS["detail"] = f"probe: {str(exc)[:120]}"
    return dict(RAG_STATUS)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.post("/local/stream")
def local_stream(request: QuestionRequest) -> StreamingResponse:
    """Pass-through proxy to the production API's SSE endpoint."""

    def _proxy() -> Iterator[bytes]:
        with httpx.Client(timeout=httpx.Timeout(None, connect=15.0)) as client:
            # The production pipeline can retry on 429 for minutes before the
            # first byte; no read timeout so the stream survives that.
            with client.stream(
                "POST",
                f"{LOCAL_API}/query/stream",
                json={"question": request.question},
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", "replace")
                    yield (
                        "event: error\ndata: "
                        + json.dumps({"detail": f"local API {resp.status_code}: {body[:200]}"})
                        + "\n\n"
                    )
                    return
                for chunk in resp.iter_bytes():
                    yield chunk

    return StreamingResponse(_proxy(), media_type="text/event-stream")


def _source_name(rc) -> str | None:
    for attr in ("title", "uri"):
        val = getattr(rc, attr, None)
        if val:
            name = str(val).rstrip("/")
            return name.split("/")[-1] or str(val)
    return None


def _rag_attempt(question: str, instant_chunks: list[dict] | None = None) -> Iterator[str]:
    """One generate_content_stream attempt; yields SSE events.

    instant_chunks=None -> grounded mode: VertexRagStore retrieval inside
    the generate call (the real RAG Engine UX).
    instant_chunks=<parsed chunks of the just-uploaded file> -> instant
    mode: the file's text goes straight into the prompt while the corpus
    import is still running.
    """
    from google.genai import types as gtypes

    if instant_chunks is not None:
        name = RAG_STATUS.get("filename") or "uploaded_document"
        first = {
            "sources": [{"filename": name, "page": None, "score": None}],
            "retrieved_chunks": len(instant_chunks),
        }
        yield f"event: sources\ndata: {json.dumps(first)}\n\n"
        picked = _pick_chunks(question, instant_chunks)
        parts = []
        for i, c in enumerate(picked, start=1):
            page = c.get("page")
            label = f"{name}, page {page}" if page else name
            parts.append(f"[{i}] ({label})\n{c['text']}")
        prompt = INSTANT_PROMPT.format(context="\n\n".join(parts), question=question)
        for chunk in _genai().models.generate_content_stream(
            model=settings.gemini_model,
            contents=prompt,
        ):
            text = chunk.text
            if text:
                yield f"event: token\ndata: {json.dumps(text)}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    config = gtypes.GenerateContentConfig(
        tools=[
            gtypes.Tool(
                retrieval=gtypes.Retrieval(
                    vertex_rag_store=gtypes.VertexRagStore(
                        rag_corpora=[settings.rag_corpus_name],
                        similarity_top_k=RAG_TOP_K,
                    )
                )
            )
        ],
    )

    sources_sent = False
    seen: dict[str, dict] = {}

    def _sources_event() -> str:
        return (
            "event: sources\ndata: "
            + json.dumps(
                {
                    "sources": list(seen.values()),
                    "retrieved_chunks": len(seen),
                }
            )
            + "\n\n"
        )

    for chunk in _genai().models.generate_content_stream(
        model=settings.gemini_model,
        contents=question,
        config=config,
    ):
        # Citations ride on the streamed chunks' grounding metadata; emit the
        # sources event the moment Google returns any (this IS the RAG UX).
        for cand in chunk.candidates or []:
            gm = cand.grounding_metadata
            for gc in getattr(gm, "grounding_chunks", None) or []:
                rc = getattr(gc, "retrieved_context", None)
                name = _source_name(rc) if rc is not None else None
                if name:
                    seen[name] = {"filename": name, "page": None, "score": None}
        if not sources_sent and seen:
            sources_sent = True
            yield _sources_event()
        text = chunk.text
        if text:
            yield f"event: token\ndata: {json.dumps(text)}\n\n"

    if not sources_sent:
        # Some grounded answers attach no citations - say so explicitly.
        yield _sources_event()
    yield "event: done\ndata: {}\n\n"


def _rag_stream(request: QuestionRequest) -> StreamingResponse:
    fname = RAG_STATUS.get("filename")
    instant = UPLOADS.get(fname) if (RAG_STATUS.get("state") == "importing" and fname) else None
    mode = "instant" if instant is not None else "grounded"

    def _with_retries() -> Iterator[str]:
        yield (
            "event: mode\ndata: "
            + json.dumps({"mode": mode})
            + "\n\n"
        )
        for attempt, delay in enumerate((0.0,) + RAG_RETRY_DELAYS):
            if delay:
                yield (
                    "event: retry\ndata: "
                    + json.dumps({"detail": f"Vertex 429; retrying in {delay:.0f}s"})
                    + "\n\n"
                )
                time.sleep(delay)
            try:
                yield from _rag_attempt(request.question, instant)
                return
            except Exception as exc:  # noqa: BLE001
                last = attempt == len(RAG_RETRY_DELAYS)
                if last:
                    yield (
                        "event: error\ndata: "
                        + json.dumps({"detail": f"RAG Engine failed: {exc}"})
                        + "\n\n"
                    )
                    return

    return StreamingResponse(_with_retries(), media_type="text/event-stream")


app.post("/rag/stream")(_rag_stream)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")