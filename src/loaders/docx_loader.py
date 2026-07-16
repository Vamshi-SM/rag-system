"""DOCX document loader using python-docx.

DOCX has no native concept of "pages" (pagination is a rendering-time
detail decided by the viewer), so ``count_pages`` intentionally returns
``None`` rather than a misleading estimate.
"""

from __future__ import annotations

from pathlib import Path

import docx

from src.loaders.base_loader import BaseLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocxLoader(BaseLoader):
    """Loader for ``.docx`` files."""

    supported_extensions = (".docx",)

    def extract_text(self, file_path: Path) -> str:
        """Extract paragraph and table text in document order."""
        document = docx.Document(str(file_path))
        blocks: list[str] = []

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                blocks.append(paragraph.text)

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))

        return "\n\n".join(blocks)
