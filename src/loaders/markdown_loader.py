"""Markdown document loader.

We keep the raw markdown syntax intact (headers, lists, etc.) rather
than stripping it to plain text, since markdown structure is often
useful signal for downstream chunking (e.g. splitting on headers).
"""

from __future__ import annotations

from pathlib import Path

from src.loaders.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
    """Loader for ``.md`` files."""

    supported_extensions = (".md",)

    def extract_text(self, file_path: Path) -> str:
        """Read the markdown file with encoding fallback handling."""
        return self._read_with_encoding_fallback(file_path)
