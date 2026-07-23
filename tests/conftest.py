"""Shared pytest fixtures for the RAG system test suite.

Fixtures here provide:

* A temporary, isolated set of sample documents (txt/md/docx/pdf) for
  loader/chunking/ingestion tests.
* A temporary SQLite database path per test, so tests never touch the
  real project database.
* Deterministic, network-free fakes for the embedding and LLM APIs
  (patched at the ``_call_api`` seam of ``QwenEmbedding`` / ``SLLMChat``),
  so the whole suite runs offline and reproducibly.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import pytest

# Allow `import src...` / `import scripts...` when pytest is run from
# the project root (also works via rootdir-based sys.path insertion,
# this is just a safety net for ad-hoc invocations).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.embeddings.qwen_embedding import QwenEmbedding
from src.embeddings.google_embedding import GoogleEmbedding
from src.llm.sllm import SLLMChat
from src.vectordb.vector_store import VectorStore

FAKE_EMBEDDING_DIM = 16


def _deterministic_vector(text: str, dim: int = FAKE_EMBEDDING_DIM) -> list[float]:
    """Produce a deterministic pseudo-embedding for a piece of text.

    Not semantically meaningful, but stable across runs (same text ->
    same vector), which is exactly what's needed to test the plumbing
    (batching, storage, similarity search, thresholding) without
    calling a real embedding API.
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


@pytest.fixture
def sample_documents_dir(tmp_path: Path) -> Path:
    """Create a temp directory with one sample file of each supported type."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    (docs_dir / "policy.txt").write_text(
        "Refunds are available within 30 days of purchase for unused subscription time. "
        "Customers should contact support to initiate a refund request.",
        encoding="utf-8",
    )

    (docs_dir / "pricing.md").write_text(
        "# Pricing\n\nThe Pro plan costs $49 per month and includes unlimited document ingestion.",
        encoding="utf-8",
    )

    docx_path = docs_dir / "handbook.docx"
    _write_sample_docx(docx_path)

    pdf_path = docs_dir / "manual.pdf"
    _write_sample_pdf(pdf_path)

    # An unsupported file type, to verify graceful skipping.
    (docs_dir / "notes.xyz").write_text("unsupported file type", encoding="utf-8")

    return docs_dir


def _write_sample_docx(path: Path) -> None:
    import docx

    document = docx.Document()
    document.add_paragraph("Employee Handbook")
    document.add_paragraph(
        "All employees are entitled to 20 days of paid time off per calendar year."
    )
    document.save(str(path))


def _write_sample_pdf(path: Path) -> None:
    from reportlab.pdfgen import canvas

    pdf_canvas = canvas.Canvas(str(path))
    pdf_canvas.drawString(72, 750, "Product Manual - Page 1")
    pdf_canvas.drawString(72, 730, "The device should be charged for 2 hours before first use.")
    pdf_canvas.showPage()
    pdf_canvas.drawString(72, 750, "Product Manual - Page 2")
    pdf_canvas.drawString(72, 730, "Warranty coverage extends for 12 months from purchase date.")
    pdf_canvas.showPage()
    pdf_canvas.save()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A fresh SQLite database path, isolated per test."""
    return tmp_path / "test_vectors.db"


@pytest.fixture
def vector_store(db_path: Path):
    """A VectorStore backed by a temporary database, closed after the test."""
    store = VectorStore(database_path=db_path, default_top_k=5)
    yield store
    store.close()


@pytest.fixture
def patch_embedding_api(monkeypatch: pytest.MonkeyPatch):
    """Patch QwenEmbedding._call_api with a deterministic, offline fake."""

    def fake_call_api(self, texts: list[str]) -> list[list[float]]:
        vectors = [_deterministic_vector(text) for text in texts]
        if vectors and getattr(self, "_dimensions", None) is None:
            self._dimensions = len(vectors[0])
        return vectors

    monkeypatch.setattr(QwenEmbedding, "_call_api", fake_call_api)
    monkeypatch.setattr(OllamaEmbedding, "_call_api", fake_call_api)
    return fake_call_api


@pytest.fixture
def patch_llm_api(monkeypatch: pytest.MonkeyPatch):
    """Patch SLLMChat._call_api with a canned, offline fake response."""

    def fake_call_api(self: SLLMChat, messages: list[dict]) -> str:
        # Echo back a short canned answer; tests assert on presence of
        # sources/behavior rather than exact LLM wording.
        return "Based on the provided context, here is the answer."

    monkeypatch.setattr(SLLMChat, "_call_api", fake_call_api)
    return fake_call_api


@pytest.fixture
def embedder(patch_embedding_api) -> QwenEmbedding:
    """A QwenEmbedding instance wired to the offline fake API."""
    return QwenEmbedding(api_key="test-key", base_url="https://fake-sllm.test/v1", model="test-embed")


@pytest.fixture
def llm(patch_llm_api) -> SLLMChat:
    """An SLLMChat instance wired to the offline fake API."""
    return SLLMChat(api_key="test-key", base_url="https://fake-sllm.test/v1", model="test-chat")


@pytest.fixture(autouse=True)
def isolate_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the global `settings` singleton at a temp directory for every test.

    Prevents any test (especially API tests, which build components from
    `settings` during FastAPI's lifespan) from touching the real project
    database or log directory.
    """
    monkeypatch.setattr(settings, "data_directory", tmp_path / "documents")
    monkeypatch.setattr(settings, "processed_directory", tmp_path / "processed")
    monkeypatch.setattr(settings, "database_path", tmp_path / "processed" / "vectors.db")
    monkeypatch.setattr(settings, "log_directory", tmp_path / "logs")
    monkeypatch.setattr(settings, "allowed_ingest_root", tmp_path)
    monkeypatch.setattr(settings, "similarity_threshold", -1.0)  # accept all in tests by default
