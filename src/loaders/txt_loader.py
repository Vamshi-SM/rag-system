"""Plain-text document loader."""

from __future__ import annotations

from pathlib import Path

from src.loaders.base_loader import BaseLoader


class TxtLoader(BaseLoader):
    """Loader for ``.txt`` files."""

    supported_extensions = (".txt",)

    def extract_text(self, file_path: Path) -> str:
        """Read the file with encoding fallback handling."""
        return self._read_with_encoding_fallback(file_path)
