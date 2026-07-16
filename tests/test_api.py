"""API tests for Phase 9 (POST /ingest, POST /query, GET /health, GET /stats).

Uses FastAPI's TestClient (sync httpx wrapper) so the app's lifespan
(startup/shutdown) runs exactly as it would under uvicorn, with the
embedding and LLM APIs patched to deterministic offline fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture
def client(patch_embedding_api, patch_llm_api) -> TestClient:
    """A TestClient for the full app, with offline fakes for embedding/LLM.

    ``isolate_settings`` (autouse, from conftest.py) already points
    ``settings`` at a temp directory before the app's lifespan builds
    its components, so this never touches the real project database.
    """
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    def test_health_returns_200_and_healthy_status(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["database"] == "connected"
        assert body["embedding_service"] == "online"
        assert body["llm"] == "online"


class TestStatsEndpoint:
    def test_stats_on_empty_store(self, client: TestClient) -> None:
        response = client.get("/stats")
        assert response.status_code == 200
        body = response.json()
        assert body["documents"] == 0
        assert body["chunks"] == 0
        assert body["embeddings"] == 0
        assert "MB" in body["database_size"]

    def test_stats_reflect_ingested_documents(
        self, client: TestClient, sample_documents_dir: Path
    ) -> None:
        client.post("/ingest", json={"folder": str(sample_documents_dir)})

        response = client.get("/stats")
        body = response.json()

        assert body["documents"] == 4
        assert body["chunks"] > 0
        assert body["embeddings"] == body["chunks"]


class TestIngestEndpoint:
    def test_ingest_success(self, client: TestClient, sample_documents_dir: Path) -> None:
        response = client.post("/ingest", json={"folder": str(sample_documents_dir)})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["documents"] == 4
        assert body["chunks"] > 0
        assert body["stored_vectors"] == body["chunks"]
        assert "sec" in body["processing_time"]

    def test_ingest_second_call_reports_all_as_duplicates(
        self, client: TestClient, sample_documents_dir: Path
    ) -> None:
        client.post("/ingest", json={"folder": str(sample_documents_dir)})

        response = client.post("/ingest", json={"folder": str(sample_documents_dir)})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["documents"] == 0  # no *new* documents
        assert body["duplicates_skipped"] == 4
        assert body["chunks"] == 0

    def test_ingest_nonexistent_folder_returns_404(
        self, client: TestClient, sample_documents_dir: Path
    ) -> None:
        missing_dir = sample_documents_dir.parent / "does-not-exist"
        response = client.post("/ingest", json={"folder": str(missing_dir)})
        assert response.status_code == 404
        assert response.json()["status"] == "error"

    def test_ingest_blank_folder_returns_422(self, client: TestClient) -> None:
        response = client.post("/ingest", json={"folder": "   "})
        assert response.status_code == 422
        assert response.json()["status"] == "error"

    def test_ingest_missing_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/ingest", json={})
        assert response.status_code == 422

    def test_ingest_invalid_json_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/ingest", content="not valid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_ingest_path_outside_allowed_root_returns_403(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        outside_dir = tmp_path.parent / "outside-allowed-root"
        outside_dir.mkdir(exist_ok=True)
        (outside_dir / "secret.txt").write_text("should not be reachable", encoding="utf-8")

        response = client.post("/ingest", json={"folder": str(outside_dir)})

        assert response.status_code == 403
        assert response.json()["status"] == "error"


class TestQueryEndpoint:
    def test_query_after_ingestion_returns_answer_and_sources(
        self, client: TestClient, sample_documents_dir: Path
    ) -> None:
        client.post("/ingest", json={"folder": str(sample_documents_dir)})

        response = client.post("/query", json={"question": "What is the refund policy?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert isinstance(body["sources"], list)
        assert body["retrieved_chunks"] >= 0
        assert "sec" in body["latency"]
        for source in body["sources"]:
            assert "filename" in source
            assert "score" in source

    def test_query_with_no_ingested_documents_returns_not_available_answer(
        self, client: TestClient
    ) -> None:
        response = client.post("/query", json={"question": "Anything?"})
        assert response.status_code == 200
        body = response.json()
        assert body["retrieved_chunks"] == 0

    def test_query_blank_question_returns_422(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "   "})
        assert response.status_code == 422

    def test_query_missing_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_top_k_out_of_range_returns_422(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "test", "top_k": 0})
        assert response.status_code == 422

    def test_query_respects_top_k(self, client: TestClient, sample_documents_dir: Path) -> None:
        client.post("/ingest", json={"folder": str(sample_documents_dir)})

        response = client.post("/query", json={"question": "policy", "top_k": 1})

        assert response.status_code == 200
        assert response.json()["retrieved_chunks"] <= 1


class TestSwaggerDocs:
    def test_openapi_json_lists_all_endpoints(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        assert set(paths.keys()) == {"/ingest", "/query", "/health", "/stats"}

    def test_docs_ui_available(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_ui_available(self, client: TestClient) -> None:
        response = client.get("/redoc")
        assert response.status_code == 200
