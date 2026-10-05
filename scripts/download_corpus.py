"""Download a 100 MB public-domain legal corpus for RAG latency benchmarking.

Streams US court opinions from the Hugging Face dataset
``wildphoton/courtlistener_opinions`` (CourtListener opinions, public
domain) and concatenates them into ~500 KB text files until the target
corpus size is reached. The ~30 real documents in ``data/documents`` are
copied alongside so ground-truth questions keep known answers.

Outputs (defaults):
  data/documents_large/           corpus files (bulk_part_NNNN.txt + real docs)
  benchmarks/corpus_manifest.json provenance: file -> case ids/sources

Run:  python scripts/download_corpus.py [--target-mb 100] [--out-dir data/documents_large]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("download_corpus")

DATASET = "wildphoton/courtlistener_opinions"
TEXT_KEY = "text"
BYTES_PER_FILE = 500 * 1024          # ~500 KB per output file
MIN_RECORD_CHARS = 1_000             # drop useless stubs
MAX_RECORD_CHARS = 400_000           # drop pathological outliers
PROGRESS_EVERY = 500


def _clean_text(text: str) -> str:
    """Collapse the newlines/spacing artifacts common in bulk opinion dumps."""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _flush(
    buffer: list[str],
    case_ids: list[str],
    sources: list[str],
    out_dir: Path,
    file_index: int,
) -> dict:
    """Write the buffered cases to the next bulk file; return its manifest entry."""
    filename = f"bulk_part_{file_index:04d}.txt"
    path = out_dir / filename
    path.write_text("\n\n".join(buffer), encoding="utf-8", newline="\n")
    entry = {
        "filename": filename,
        "bytes": path.stat().st_size,
        "n_cases": len(case_ids),
        "case_ids": case_ids,
        "sources": sorted(set(sources)),
    }
    logger.info(
        "Wrote %s (%.1f KB, %d cases)", filename, entry["bytes"] / 1024, len(case_ids)
    )
    return entry


def download_corpus(target_mb: float, out_dir: Path, manifest_path: Path) -> dict:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing_bulk = sorted(out_dir.glob("bulk_part_*.txt"))
    if existing_bulk:
        logger.warning(
            "Found %d existing bulk files in %s; starting numbering after them.",
            len(existing_bulk), out_dir,
        )

    target_bytes = int(target_mb * 1024 * 1024)
    copied_real_docs = [
        p.name for p in (REPO_ROOT / "data" / "documents").iterdir()
        if p.is_file() and p.name != ".gitkeep"
    ]
    for name in copied_real_docs:
        shutil.copy2(REPO_ROOT / "data" / "documents" / name, out_dir / name)
    logger.info("Copied %d real documents into %s", len(copied_real_docs), out_dir)

    file_index = len(existing_bulk) + 1
    written = 0
    n_cases = 0
    seen_ids: set[str] = set()
    buffer: list[str] = []
    case_ids: list[str] = []
    sources: list[str] = []
    manifest_files: list[dict] = []
    skipped_short = skipped_dup = 0

    stream = load_dataset(DATASET, split="train", streaming=True)
    for record in stream:
        case_id = str(record.get("id", ""))
        if case_id in seen_ids:
            skipped_dup += 1
            continue
        text = _clean_text(record.get(TEXT_KEY) or "")
        if len(text) < MIN_RECORD_CHARS:
            skipped_short += 1
            continue
        seen_ids.add(case_id)

        source = str(record.get("source", "") or "unknown")
        created = str(record.get("created", "") or "")
        header = f"=== CASE {case_id} | SOURCE {source} | CREATED {created} ==="
        chunk_text = f"{header}\n\n{text}"

        if len(chunk_text) > MAX_RECORD_CHARS:
            chunk_text = chunk_text[:MAX_RECORD_CHARS]

        buffer.append(chunk_text)
        case_ids.append(case_id)
        sources.append(source)
        written += len(chunk_text.encode("utf-8"))
        n_cases += 1

        if sum(len(b) for b in buffer) >= BYTES_PER_FILE:
            manifest_files.append(_flush(buffer, case_ids, sources, out_dir, file_index))
            buffer, case_ids, sources = [], [], []
            file_index += 1

        if n_cases % PROGRESS_EVERY == 0:
            logger.info(
                "Progress: %d cases, %.1f / %d MB", n_cases, written / (1024 * 1024), target_mb
            )
        if written >= target_bytes:
            break

    if buffer:
        manifest_files.append(_flush(buffer, case_ids, sources, out_dir, file_index))

    manifest = {
        "dataset": DATASET,
        "target_mb": target_mb,
        "total_bytes": sum(f["bytes"] for f in manifest_files),
        "total_cases": n_cases,
        "skipped_short": skipped_short,
        "skipped_duplicates": skipped_dup,
        "bulk_files": manifest_files,
        "real_documents": copied_real_docs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-mb", type=float, default=100.0)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / "documents_large"
    )
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "benchmarks" / "corpus_manifest.json"
    )
    args = parser.parse_args()

    manifest = download_corpus(args.target_mb, args.out_dir, args.manifest)
    total_mb = manifest["total_bytes"] / (1024 * 1024)
    print("\n================ DOWNLOAD SUMMARY ================")
    print(f"Corpus size     : {total_mb:.1f} MB in {len(manifest['bulk_files'])} bulk files")
    print(f"Opinions        : {manifest['total_cases']:,}")
    print(f"Real docs kept  : {len(manifest['real_documents'])}")
    print(f"Skipped         : {manifest['skipped_short']} short, "
          f"{manifest['skipped_duplicates']} duplicate")
    print(f"Output dir      : {args.out_dir}")
    print(f"Manifest        : {args.manifest}")
    print("==================================================")
    if total_mb < args.target_mb * 0.98:
        print("WARNING: corpus fell short of target - dataset may be exhausted.")
        sys.exit(2)


if __name__ == "__main__":
    main()