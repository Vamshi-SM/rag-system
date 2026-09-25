"""API routes: POST /ingest, POST /query, POST /query/latency, GET /health,
GET /stats.

All business logic is delegated to existing Phase 2-8 modules
(``IngestionService``, ``RAGPipeline``, ``Retriever``) - these routes are
thin: they validate input, call the appropriate service, and shape the
response.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from src.api.dependencies import (
    get_embedder,
    get_ingestion_service,
    get_llm,
    get_rag_pipeline,
    get_retriever,
    get_vector_store,
)
from src.api.exceptions import (
    FolderNotFoundError,
    NoDocumentsFoundError,
    PathNotAllowedError,
    ServiceUnavailableError,
)
from src.api.schemas import (
    ChunkModel,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LatencyRequest,
    LatencyResponse,
    QueryRequest,
    QueryResponse,
    SourceModel,
    StageTiming,
    StatsResponse,
)
from src.config import settings
from src.embeddings.base_embedding import BaseEmbedding
from src.ingestion.ingestion_service import IngestionService
from src.llm.base_llm import BaseLLM
from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.retriever import RetrievalError, Retriever, RetrievedChunk
from src.utils.logger import get_logger
from src.utils.profiler import Profiler, use_profiler
from src.vectordb.vector_store import VectorStore

logger = get_logger(__name__)

router = APIRouter()


def _format_duration(seconds: float) -> str:
    return f"{seconds:.2f} sec"


@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingest documents from a folder",
    description=(
        "Recursively scans the given folder for supported documents (PDF, DOCX, "
        "TXT, Markdown), skips duplicates already ingested, chunks and embeds "
        "the rest, and stores them in the SQLite + sqlite-vec vector database."
    ),
    responses={
        403: {"description": "Folder is outside the allowed ingest root"},
        404: {"description": "Folder not found"},
        422: {"description": "Validation error (e.g. blank folder path)"},
    },
)
async def ingest_documents(
    request: IngestRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Ingest all supported documents in ``request.folder``."""
    folder = Path(request.folder).resolve()
    allowed_root = settings.allowed_ingest_root.resolve()

    if not folder.is_relative_to(allowed_root):
        raise PathNotAllowedError(request.folder, str(allowed_root))

    if not folder.exists() or not folder.is_dir():
        raise FolderNotFoundError(request.folder)

    logger.info("Ingestion requested for folder '%s'", folder)
    start_time = time.time()

    # Ingestion does blocking I/O (file reads, embedding HTTP calls,
    # SQLite writes) - run it off the event loop so the API stays
    # responsive to other requests while it works.
    result = await run_in_threadpool(ingestion_service.run, folder)

    if result.documents_found == 0:
        raise NoDocumentsFoundError(request.folder)

    elapsed = time.time() - start_time
    logger.info(
        "Ingestion finished for '%s': %d new documents, %d chunks, %d vectors (%.2fs)",
        folder,
        result.new_documents,
        result.chunks_created,
        result.vectors_stored,
        elapsed,
    )

    return IngestResponse(
        status=result.status,
        documents=result.new_documents,
        chunks=result.chunks_created,
        stored_vectors=result.vectors_stored,
        duplicates_skipped=result.duplicates_skipped,
        processing_time=_format_duration(elapsed),
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Ask a question against the ingested documents",
    description=(
        "Embeds the question, retrieves the top-K most relevant chunks from the "
        "vector store, builds a context-grounded prompt, and calls the LLM to "
        "generate an answer. Never hallucinates beyond the retrieved context."
    ),
    responses={422: {"description": "Validation error (e.g. blank question)"}},
)
async def query_documents(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """Answer ``request.question`` using retrieval-augmented generation."""
    logger.info("Query requested: '%s'", request.question)

    response = await run_in_threadpool(
        rag_pipeline.answer,
        request.question,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        filename=request.filename,
    )

    # response.sources is de-duplicated (Phase 8's build_sources groups by
    # filename+page), but chunks_used is the raw retrieved list with one
    # score per chunk - build sources straight from chunks_used here so
    # filename/page/score always line up correctly.
    sources = [
        SourceModel(
            filename=chunk["metadata"].get("filename", "unknown"),
            page=chunk["metadata"].get("page"),
            score=chunk["score"],
        )
        for chunk in response.chunks_used
    ]

    return QueryResponse(
        answer=response.answer,
        sources=sources,
        retrieved_chunks=len(response.chunks_used),
        latency=_format_duration(response.elapsed_seconds),
    )


@router.post(
    "/query/latency",
    response_model=LatencyResponse,
    tags=["Query"],
    summary="Measure retrieval-only latency (no LLM call)",
    description=(
        "Runs the retrieval half of the pipeline only - query embedding, "
        "vector search, keyword search, RRF fusion, and filtering/ranking. "
        "The prompt is never built and the LLM is never called, so the "
        "reported timings isolate exactly how long a query spends before "
        "answer generation. Set ``runs`` (up to 25) to repeat the same "
        "question and get avg/min/max plus a per-stage breakdown."
    ),
    responses={
        422: {"description": "Validation error (e.g. blank question, out-of-range runs)"},
        503: {"description": "Embedding service or vector store unreachable"},
    },
)
async def query_latency(
    request: LatencyRequest,
    retriever: Retriever = Depends(get_retriever),
) -> LatencyResponse:
    """Benchmark retrieval-only latency for ``request.question``."""

    def _benchmark() -> tuple[Profiler, list[float], list[RetrievedChunk]]:
        # A single Profiler scopes all runs; per-stage entries accumulate,
        # so each StageTiming reports the average across ``runs`` passes.
        profiler = Profiler()
        run_times: list[float] = []
        chunks: list[RetrievedChunk] = []
        with use_profiler(profiler):
            for _ in range(request.runs):
                run_start = time.perf_counter()
                chunks = retriever.retrieve_with_scores(
                    request.question,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    filename=request.filename,
                )
                run_times.append(time.perf_counter() - run_start)
        return profiler, run_times, chunks

    logger.info(
        "Latency test requested: '%s' (runs=%d, top_k=%s)",
        request.question,
        request.runs,
        request.top_k,
    )

    try:
        profiler, run_times, chunks = await run_in_threadpool(_benchmark)
    except RetrievalError as exc:
        logger.error("Latency test failed (retrieval error): %s", exc)
        raise ServiceUnavailableError(f"Retrieval failed: {exc}") from exc

    avg_seconds = sum(run_times) / len(run_times)
    stages = [
        StageTiming(
            stage=name,
            milliseconds=entry["seconds"] / entry["count"] * 1000.0,
            calls=int(entry["count"]),
        )
        for name, entry in profiler.stages.items()
    ]

    sources = [
        SourceModel(
            filename=chunk["metadata"].get("filename", "unknown"),
            page=chunk["metadata"].get("page"),
            score=chunk["score"],
        )
        for chunk in chunks
    ]

    retrieved_chunks_full = [
        ChunkModel(
            filename=chunk["metadata"].get("filename", "unknown"),
            page=chunk["metadata"].get("page"),
            score=chunk["score"],
            text=chunk["text"],
        )
        for chunk in chunks
    ]

    logger.info(
        "Latency test finished: %d runs, avg %.1fms, %d chunks retrieved",
        len(run_times),
        avg_seconds * 1000.0,
        len(chunks),
    )

    return LatencyResponse(
        runs=len(run_times),
        retrieved_chunks=len(chunks),
        top_score=chunks[0]["score"] if chunks else None,
        sources=sources,
        chunks=retrieved_chunks_full,
        run_times_ms=[t * 1000.0 for t in run_times],
        avg_ms=avg_seconds * 1000.0,
        min_ms=min(run_times) * 1000.0,
        max_ms=max(run_times) * 1000.0,
        stages=stages,
        latency=_format_duration(avg_seconds),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check",
    description="Reports whether the database, embedding service, and LLM are reachable.",
)
async def health_check(
    vector_store: VectorStore = Depends(get_vector_store),
    embedder: BaseEmbedding = Depends(get_embedder),
    llm: BaseLLM = Depends(get_llm),
) -> HealthResponse:
    """Check connectivity to the database, embedding API, and LLM API."""

    def _check_database() -> str:
        try:
            vector_store.count()
            return "connected"
        except Exception as exc:  # noqa: BLE001
            logger.error("Health check: database unreachable: %s", exc)
            return "disconnected"

    def _check_embedding_service() -> str:
        try:
            embedder.embed("healthcheck")
            return "online"
        except Exception as exc:  # noqa: BLE001 - health check must never crash
            logger.error("Health check: embedding service unreachable: %s", exc)
            return "offline"

    def _check_llm() -> str:
        try:
            llm.generate("respond with OK")
            return "online"
        except Exception as exc:  # noqa: BLE001 - health check must never crash
            logger.error("Health check: LLM unreachable: %s", exc)
            return "offline"

    database_status, embedding_status, llm_status = await run_in_threadpool(
        lambda: (_check_database(), _check_embedding_service(), _check_llm())
    )

    overall = (
        "healthy"
        if database_status == "connected"
        and embedding_status == "online"
        and llm_status == "online"
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        database=database_status,
        embedding_service=embedding_status,
        llm=llm_status,
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Monitoring"],
    summary="Vector store statistics",
    description="Returns document/chunk/embedding counts and the on-disk database size.",
)
async def get_stats(
    vector_store: VectorStore = Depends(get_vector_store),
) -> StatsResponse:
    """Return ingestion statistics and database size."""

    def _collect_stats() -> tuple[int, int, int]:
        documents = len(vector_store.list_ingested_documents())
        chunks = vector_store.count()
        return documents, chunks, chunks  # 1 embedding per stored chunk

    documents, chunks, embeddings = await run_in_threadpool(_collect_stats)

    db_path = Path(vector_store.database.database_path)
    size_bytes = db_path.stat().st_size if db_path.exists() else 0
    database_size = f"{size_bytes / (1024 * 1024):.1f} MB"

    return StatsResponse(
        documents=documents,
        chunks=chunks,
        embeddings=embeddings,
        database_size=database_size,
    )
