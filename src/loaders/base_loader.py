"""Common interface and shared helpers for all document loaders.

Every concrete loader (PDF, DOCX, TXT, Markdown) implements
``BaseLoader`` and returns documents in the exact same shape so that
downstream chunking/embedding code never needs to know which loader
produced a given document.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentMetadata(TypedDict, total=False):
    """Metadata attached to every loaded document."""

    extension: str
    pages: Optional[int]
    created_at: str
    source: str
    page_texts: Optional[list[str]]


class LoadedDocument(TypedDict):
    """The canonical shape returned by every loader."""

    id: str
    filename: str
    text: str
    metadata: DocumentMetadata


class DocumentLoadError(Exception):
    """Raised when a file cannot be parsed by a loader."""


class BaseLoader(ABC):
    """Abstract base class defining the loader contract.

    Subclasses only need to implement :meth:`extract_text` (and
    optionally override :meth:`count_pages`); :meth:`load` handles all
    the shared bookkeeping (id generation, metadata, empty-doc
    filtering, error handling).
    """

    #: File extensions (lowercase, including the dot) this loader supports.
    supported_extensions: tuple[str, ...] = ()

    def supports(self, file_path: Path) -> bool:
        """Return True if this loader can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        """Extract raw text content from the file, preserving order."""
        raise NotImplementedError

    def count_pages(self, file_path: Path) -> Optional[int]:
        """Return a page count if meaningful for this file type, else None."""
        return None

    def extract_page_texts(self, file_path: Path) -> Optional[list[str]]:
        """Return per-page text if pagination is meaningful, else None.

        Only overridden by loaders (e.g. PDF) where "page" is a real,
        stable unit. Returning None tells the chunker to treat the
        whole document as one continuous stream of text.
        """
        return None

    def load(self, file_path: Path) -> Optional[LoadedDocument]:
        """Load a single file into the canonical document schema.

        Returns:
            A ``LoadedDocument`` dict, or ``None`` if the document was
            empty (and therefore intentionally skipped).

        Raises:
            DocumentLoadError: If the file could not be parsed at all.
        """
        try:
            text = self.extract_text(file_path)
        except Exception as exc:  # noqa: BLE001 - we deliberately wrap all errors
            logger.error("Failed to load '%s': %s", file_path, exc)
            raise DocumentLoadError(f"Could not load {file_path}: {exc}") from exc

        if not text or not text.strip():
            logger.warning("Skipping empty document: %s", file_path)
            return None

        stat = file_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()

        document: LoadedDocument = {
            "id": str(uuid.uuid4()),
            "filename": file_path.name,
            "text": text,
            "metadata": {
                "extension": file_path.suffix.lower(),
                "pages": self.count_pages(file_path),
                "created_at": created_at,
                "source": str(file_path.resolve()),
                "page_texts": self.extract_page_texts(file_path),
            },
        }
        logger.info("Loaded document '%s' (%d chars)", file_path.name, len(text))
        return document

    def _read_with_encoding_fallback(self, file_path: Path) -> str:
        """Read a text file, gracefully handling encoding issues.

        Tries UTF-8 first, then falls back to latin-1, and finally
        decodes with errors replaced so a single bad byte never crashes
        an entire ingestion run.
        """
        encodings = ("utf-8", "utf-8-sig", "latin-1")
        last_error: Optional[UnicodeDecodeError] = None

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue

        logger.warning(
            "Encoding fallback exhausted for '%s' (%s); decoding with errors replaced",
            file_path,
            last_error,
        )
        raw_bytes = file_path.read_bytes()
        return raw_bytes.decode("utf-8", errors="replace")

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.__class__.__name__}(extensions={self.supported_extensions})"
