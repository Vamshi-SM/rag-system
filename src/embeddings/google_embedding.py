from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai.types import EmbedContentConfig
from tqdm import tqdm

from src.chunking.chunker import Chunk
from src.embeddings.base_embedding import (
    BaseEmbedding,
    EmbeddedChunk,
)
from src.utils.logger import get_logger
from google.genai.errors import ClientError
logger = get_logger(__name__)


class GoogleEmbedding(BaseEmbedding):

    def __init__(
        self,
        project_id: str,
        location: str = "asia-south1",
        model: str = "text-embedding-004",
        batch_size: int = 8,
        max_chars: int = 20000,
        max_request_tokens: int = 19000,
        concurrency: int = 1,
        progress_cb=None,
    ):
        self.project_id = project_id
        self.location = location
        self.model = model
        self.batch_size = batch_size
        # Secondary payload guard in characters. The binding constraint is
        # the model's TOTAL per-request input token cap (text-embedding-004:
        # 20,000 tokens) enforced via max_request_tokens below.
        self.max_chars = max_chars
        # Token budget per API request, enforced with tiktoken (cl100k_base,
        # same estimator as the chunker). Google's own tokenizer differs
        # slightly; 19000 of 20000 leaves that margin.
        self.max_request_tokens = max_request_tokens
        # Number of API requests in flight at once (EMBEDDING_CONCURRENCY).
        self.concurrency = max(1, int(concurrency))
        # Optional callback(done, total) fired after every completed batch,
        # used by scripts/ingest_with_progress.py for PROGRESS log lines.
        self.progress_cb = progress_cb
        self._dimensions = None
        self._skipped_chunks = 0
        self._token_len = self._build_token_counter()
        # The genai.Client resolves credentials at construction time; build
        # it lazily so the API can start (and serve LLM/embedding-free
        # routes like /query/latency and /stats) before GCP credentials
        # are configured.
        self.client = None
        # Client construction must happen exactly once: racing
        # genai.Client() from worker threads leaves a closed httpx client
        # ("Cannot send a request, as the client has been closed").
        self._client_lock = threading.Lock()

    @staticmethod
    def _build_token_counter():
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(encoding.encode(text))
        except Exception as exc:  # noqa: BLE001
            logger.warning("tiktoken unavailable (%s); estimating 4 chars/token", exc)
            return lambda text: max(1, len(text) // 4)

    def _ensure_client(self):
        if self.client is None:
            with self._client_lock:
                if self.client is None:
                    self.client = genai.Client(
                        vertexai=True,
                        project=self.project_id,
                        location=self.location,
                    )
        return self.client

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    @property
    def skipped_chunks(self) -> int:
        return self._skipped_chunks

    def _embed_with_retry(
        self, contents: str | list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ):
        """Execute embed_content with exponential backoff on 429 / Rate Limits."""
        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                return self._ensure_client().models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=EmbedContentConfig(
                        task_type=task_type,
                    ),
                )
            except ClientError as e:
                err_msg = str(e)
                if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limit / 429 hit. Retrying batch in %.1fs (attempt %d/%d)...",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                else:
                    raise

    def embed(self, text: str) -> list[float]:
        # Queries use RETRIEVAL_QUERY; text-embedding-004/005 optimize the
        # two sides differently and mixing them degrades ranking quality.
        # Ingestion goes through embed_documents/_embed_batch, which keeps
        # task_type="RETRIEVAL_DOCUMENT".
        response = self._embed_with_retry(contents=text, task_type="RETRIEVAL_QUERY")

        vector = response.embeddings[0].values

        if self._dimensions is None:
            self._dimensions = len(vector)

        return vector

    def _group_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Split chunks into API requests honoring batch_size, max_chars
        and the per-request token budget."""
        batches: list[list[Chunk]] = []
        current_batch: list[Chunk] = []
        current_chars = 0
        current_tokens = 0

        for chunk in chunks:
            text = chunk["text"]
            chars = len(text)
            tokens = self._token_len(text)

            # A single chunk larger than the budget can never be batched
            # with anything; it travels alone (and is split by the
            # overflow handler below if the API still rejects it).
            if current_batch and (
                len(current_batch) >= self.batch_size
                or current_chars + chars > self.max_chars
                or current_tokens + tokens > self.max_request_tokens
            ):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
                current_tokens = 0

            current_batch.append(chunk)
            current_chars += chars
            current_tokens += tokens

        if current_batch:
            batches.append(current_batch)
        return batches

    def _embed_batch(self, batch: list[Chunk]) -> list | None:
        """Embed one batch, returning (chunk, vector) pairs.

        Splits recursively if the API rejects the request for exceeding
        the total token limit (tokenizer drift between tiktoken and
        Google's own tokenizer). A single chunk that still fails on a
        token/argument error is skipped and counted, never fatal.
        """
        texts = [chunk["text"] for chunk in batch]
        try:
            response = self._embed_with_retry(
                contents=texts, task_type="RETRIEVAL_DOCUMENT"
            )
        except ClientError as e:
            err = str(e)
            if len(batch) > 1 and (
                "token count" in err or "INVALID_ARGUMENT" in err
            ):
                mid = len(batch) // 2
                logger.warning(
                    "Request exceeded token limit (%d chunks); splitting into %d + %d",
                    len(batch), mid, len(batch) - mid,
                )
                left = self._embed_batch(batch[:mid])
                right = self._embed_batch(batch[mid:])
                return (left or []) + (right or [])
            if "token count" in err or "INVALID_ARGUMENT" in err:
                self._skipped_chunks += 1
                logger.error(
                    "Chunk '%s' rejected by API (%s); skipping",
                    batch[0]["chunk_id"], err[:160],
                )
                return None
            raise

        vectors = [emb.values for emb in response.embeddings]
        return list(zip(batch, vectors))

    def _build_embedded(self, pairs) -> list[EmbeddedChunk]:
        if pairs and self._dimensions is None:
            self._dimensions = len(pairs[0][1])

        return [
            {
                "text": chunk["text"],
                "embedding": vector,
                "metadata": {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    **chunk["metadata"],
                },
            }
            for chunk, vector in pairs
        ]

    def _notify(self, done: int, total: int, t0: float) -> None:
        if self.progress_cb is None:
            return
        elapsed = max(time.perf_counter() - t0, 1e-6)
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0
        self.progress_cb(done, total, elapsed, rate, eta)

    def embed_documents(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:

        if not chunks:
            return []

        embedded_chunks: list[EmbeddedChunk] = []

        logger.info(
            "Embedding %d chunks using Google %s (batch_size=%d, concurrency=%d)",
            len(chunks),
            self.model,
            self.batch_size,
            self.concurrency,
        )

        batches = self._group_batches(chunks)
        logger.info("Grouped %d chunks into %d batches", len(chunks), len(batches))

        # Build the client on the calling thread before any worker touches it.
        self._ensure_client()

        total = len(chunks)
        t0 = time.perf_counter()
        skipped_baseline = self._skipped_chunks

        if self.concurrency == 1:
            for batch in tqdm(
                batches,
                desc="Embedding chunks",
                unit="batch",
            ):
                pairs = self._embed_batch(batch)
                if pairs:
                    embedded_chunks.extend(self._build_embedded(pairs))
                self._notify(
                    len(embedded_chunks) + self._skipped_chunks - skipped_baseline,
                    total, t0,
                )
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self._embed_batch, batch): batch for batch in batches
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Embedding chunks",
                    unit="batch",
                ):
                    pairs = future.result()
                    if pairs:
                        embedded_chunks.extend(self._build_embedded(pairs))
                    self._notify(
                        len(embedded_chunks) + self._skipped_chunks - skipped_baseline,
                        total, t0,
                    )

        if self._skipped_chunks > skipped_baseline:
            logger.warning(
                "%d chunk(s) skipped by the embedding API in this call",
                self._skipped_chunks - skipped_baseline,
            )

        logger.info(
            "Generated %d embeddings (dimensions=%s)",
            len(embedded_chunks),
            self._dimensions,
        )

        return embedded_chunks