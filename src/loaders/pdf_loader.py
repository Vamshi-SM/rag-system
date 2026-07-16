"""PDF document loader.

Primary extraction uses ``pdfplumber`` for high-fidelity text layout.
If ``pdfplumber`` fails to open a malformed PDF, we fall back to
``PyPDF2`` so a single problematic file doesn't halt ingestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pdfplumber

from src.loaders.base_loader import BaseLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    """Loader for ``.pdf`` files."""

    supported_extensions = (".pdf",)

    def extract_text(self, file_path: Path) -> str:
        """Extract text from a PDF, page by page, preserving order."""
        try:
            return self._extract_with_pdfplumber(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pdfplumber failed for '%s' (%s); falling back to PyPDF2",
                file_path,
                exc,
            )
            return self._extract_with_pypdf2(file_path)

    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        page_texts: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_texts.append(page_text)
                else:
                    logger.debug(
                        "No extractable text on page %d of '%s'", page_number, file_path
                    )
        return "\n\n".join(page_texts)

    def _extract_with_pypdf2(self, file_path: Path) -> str:
        # Imported lazily: PyPDF2 is only needed as a fallback path.
        import PyPDF2

        page_texts: list[str] = []
        with file_path.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_texts.append(page_text)
        return "\n\n".join(page_texts)

    def count_pages(self, file_path: Path) -> Optional[int]:
        try:
            with pdfplumber.open(file_path) as pdf:
                return len(pdf.pages)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not count pages for '%s': %s", file_path, exc)
            return None

    def extract_page_texts(self, file_path: Path) -> Optional[list[str]]:
        """Return the text of each page individually so chunks can be
        tagged with an accurate page number instead of the document's
        total page count.
        """
        try:
            page_texts: list[str] = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_texts.append(page.extract_text() or "")
            return page_texts
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not extract per-page text for '%s': %s", file_path, exc)
            return None
