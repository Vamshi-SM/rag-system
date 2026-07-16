"""Top-level orchestrator that loads an entire directory of documents.

``DocumentLoader`` is the only class the rest of the pipeline needs to
know about for Phase 2 - it dispatches each file to the correct
concrete loader (PDF/DOCX/TXT/MD) and gracefully skips anything
unsupported or empty.
"""

from __future__ import annotations

from pathlib import Path

from src.loaders.base_loader import BaseLoader, DocumentLoadError, LoadedDocument
from src.loaders.docx_loader import DocxLoader
from src.loaders.markdown_loader import MarkdownLoader
from src.loaders.pdf_loader import PDFLoader
from src.loaders.txt_loader import TxtLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Recursively loads all supported documents from a directory.

    Args:
        loaders: Optional list of loader instances to use instead of the
            default set. Useful for testing or extending with new
            file-type support without modifying this class (Open/Closed
            Principle).
    """

    def __init__(self, loaders: list[BaseLoader] | None = None) -> None:
        self.loaders: list[BaseLoader] = loaders or [
            PDFLoader(),
            DocxLoader(),
            TxtLoader(),
            MarkdownLoader(),
        ]

    def _find_loader(self, file_path: Path) -> BaseLoader | None:
        for loader in self.loaders:
            if loader.supports(file_path):
                return loader
        return None

    def load_file(self, file_path: Path) -> LoadedDocument | None:
        """Load a single file, or return None if unsupported/empty."""
        loader = self._find_loader(file_path)
        if loader is None:
            logger.debug("Skipping unsupported file type: %s", file_path)
            return None

        try:
            return loader.load(file_path)
        except DocumentLoadError as exc:
            logger.error("Skipping unreadable file '%s': %s", file_path, exc)
            return None

    def load_directory(self, directory: Path) -> list[LoadedDocument]:
        """Recursively load every supported document under ``directory``.

        Args:
            directory: Root directory to scan. Searched recursively.

        Returns:
            A list of successfully loaded documents. Unsupported,
            empty, or unreadable files are skipped and logged, never
            raised, so one bad file can't abort an entire ingestion run.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Data directory does not exist: {directory}")

        documents: list[LoadedDocument] = []
        all_files = sorted(p for p in directory.rglob("*") if p.is_file())

        logger.info("Scanning %d files under '%s'", len(all_files), directory)

        for file_path in all_files:
            document = self.load_file(file_path)
            if document is not None:
                documents.append(document)

        logger.info("Successfully loaded %d/%d documents", len(documents), len(all_files))
        return documents
