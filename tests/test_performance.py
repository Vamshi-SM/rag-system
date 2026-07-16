"""Performance benchmarks (Phase 10).

Measures ingestion throughput, embedding latency, retrieval latency,
and API response time against the offline fakes (see
``tests/conftest.py``), then prints a benchmark report.

These are not correctness assertions in the strict sense - they run
against fakes, so absolute numbers mostly reflect local Python/SQLite
overhead rather than real network latency. They exist to (a) catch
gross regressions (e.g. an accidental O(n^2) loop) and (b) give a
readable report shape that's easy to re-point at a real SLLM endpoint
for genuine production benchmarking.

Run in isolation with:

    pytest tests/test_performance.py -v -s
"""

from __future__ import annotations

import time
from pathlib import Path
from statistics import mean

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from src.api.main import create_app
from src.embeddings.qwen_embedding import QwenEmbedding
from src.ingestion.ingestion_service import IngestionService
from src.retrieval.retriever import Retriever
from src.vectordb.vector_store import VectorStore


def _make_pdfs(directory: Path, count: int) -> None:
    for index in range(count):
        path = directory / f"doc_{index}.pdf"
        pdf = canvas.Canvas(str(path))
        pdf.drawString(72, 750, f"Sample document number {index}")
        pdf.drawString(72, 730, f"This document discusses topic area {index} in some detail. " * 3)
        pdf.showPage()
        pdf.save()


class TestIngestionPerformance:
    def test_ingestion_throughput_benchmark(
        self, tmp_path: Path, vector_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        pdf_dir = tmp_path / "bench_pdfs"
        pdf_dir.mkdir()
        _make_pdfs(pdf_dir, count=10)

        service = IngestionService(vector_store=vector_store, embedder=embedder)

        start = time.perf_counter()
        result = service.run(pdf_dir)
        elapsed = time.perf_counter() - start

        docs_per_sec = result.new_documents / elapsed if elapsed > 0 else float("inf")
        chunks_per_sec = result.chunks_created / elapsed if elapsed > 0 else float("inf")

        print("\n--- Ingestion Performance ---")
        print(f"Documents ingested : {result.new_documents}")
        print(f"Chunks created     : {result.chunks_created}")
        print(f"Total time          : {elapsed:.3f}s")
        print(f"Throughput          : {docs_per_sec:.1f} docs/sec, {chunks_per_sec:.1f} chunks/sec")

        assert result.new_documents == 10
        assert elapsed < 30.0  # generous ceiling; catches gross regressions


class TestEmbeddingPerformance:
    def test_embedding_latency_benchmark(self, embedder: QwenEmbedding) -> None:
        sample_texts = [f"Sample text number {i} for latency measurement." for i in range(50)]
        latencies = []

        for text in sample_texts:
            start = time.perf_counter()
            embedder.embed(text)
            latencies.append(time.perf_counter() - start)

        avg_ms = mean(latencies) * 1000
        p95_ms = sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000

        print("\n--- Embedding Latency ---")
        print(f"Requests : {len(latencies)}")
        print(f"Average  : {avg_ms:.2f}ms")
        print(f"P95      : {p95_ms:.2f}ms")

        assert avg_ms < 1000  # fakes should be fast; catches accidental sleeps/retries


class TestRetrievalPerformance:
    @pytest.fixture
    def populated_store(self, vector_store: VectorStore, embedder: QwenEmbedding) -> VectorStore:
        chunks = []
        for i in range(200):
            text = f"Document chunk number {i} about topic {i % 10}."
            embedding = embedder.embed(text)
            chunks.append(
                {
                    "text": text,
                    "embedding": embedding,
                    "metadata": {
                        "chunk_id": f"c{i}",
                        "document_id": f"d{i % 10}",
                        "id": f"c{i}",
                        "filename": f"d{i % 10}.txt",
                        "page": None,
                    },
                }
            )
        vector_store.insert_many(chunks)
        return vector_store

    def test_retrieval_latency_benchmark(
        self, populated_store: VectorStore, embedder: QwenEmbedding
    ) -> None:
        retriever = Retriever(embedder, populated_store, default_top_k=5, default_similarity_threshold=-1.0)
        questions = [f"What about topic {i}?" for i in range(20)]
        latencies = []

        for question in questions:
            start = time.perf_counter()
            retriever.retrieve_with_scores(question)
            latencies.append(time.perf_counter() - start)

        avg_ms = mean(latencies) * 1000
        print("\n--- Retrieval Latency (200 stored chunks) ---")
        print(f"Queries  : {len(latencies)}")
        print(f"Average  : {avg_ms:.2f}ms")

        assert avg_ms < 2000


class TestAPIPerformance:
    def test_query_endpoint_response_time_benchmark(
        self, patch_embedding_api, patch_llm_api, sample_documents_dir: Path
    ) -> None:
        app = create_app()
        with TestClient(app) as client:
            client.post("/ingest", json={"folder": str(sample_documents_dir)})

            latencies = []
            for _ in range(10):
                start = time.perf_counter()
                response = client.post("/query", json={"question": "What is the refund policy?"})
                latencies.append(time.perf_counter() - start)
                assert response.status_code == 200

        avg_ms = mean(latencies) * 1000
        print("\n--- POST /query API Response Time ---")
        print(f"Requests : {len(latencies)}")
        print(f"Average  : {avg_ms:.2f}ms")

        assert avg_ms < 5000


def test_print_benchmark_summary_header() -> None:
    """Not a real assertion - just prints a readable section header so
    ``pytest -s`` output reads as a coherent benchmark report when all
    tests in this module run together.
    """
    print("\n" + "=" * 60)
    print("RAG SYSTEM PERFORMANCE BENCHMARK REPORT")
    print("=" * 60)
