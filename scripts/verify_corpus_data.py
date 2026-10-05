"""Verify that CourtListener opinion data (Hugging Face) is usable for
RAG latency benchmarking before committing to a 100 MB download.

Streams a sample of records, computes quality/suitability statistics,
prints excerpts, and estimates how many records the 100 MB corpus needs.

Run:  python scripts/verify_corpus_data.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from statistics import median

from datasets import load_dataset

TARGET_MB = 100
SAMPLE_N = 200
CHUNK_SIZE = 700

LEGAL_MARKERS = [
    " v. ", "section", "court", "appeal", "plaintiff", "defendant",
    "statute", "regulation", "contract", "opinion", "judgment",
]


def quality_flags(text: str) -> dict:
    """Cheap heuristics for OCR garbage / unusable records."""
    total = len(text)
    if total == 0:
        return {"empty": True}
    printable = sum(
        1 for c in text if unicodedata.category(c)[0] in ("L", "N", "P", "Z", "S")
    ) / total
    alpha = sum(1 for c in text if c.isalpha()) / total
    long_words = len(re.findall(r"\b[a-zA-Z]{3,}\b", text))
    words = len(re.findall(r"\S+", text))
    # A high ratio of nonsense long tokens (>= 25 chars) suggests broken OCR.
    garbage_tokens = len(re.findall(r"\b\w{25,}\b", text))
    return {
        "chars": total,
        "words": words,
        "printable_ratio": round(printable, 3),
        "alpha_ratio": round(alpha, 3),
        "long_word_ratio": round(long_words / max(words, 1), 3),
        "garbage_token_ratio": round(garbage_tokens / max(words, 1), 4),
        "legal_markers": sum(1 for m in LEGAL_MARKERS if m in text.lower()),
    }


def main() -> None:
    print(f"Streaming sample from wildphoton/courtlistener_opinions (n={SAMPLE_N})...")
    try:
        stream = load_dataset(
            "wildphoton/courtlistener_opinions", split="train", streaming=True
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to open dataset: {exc}")
        sys.exit(1)

    records = []
    text_key = None
    for record in stream:
        if text_key is None:
            print(f"\nSchema: {list(record.keys())}")
            # prefer the obvious text field
            for candidate in ("text", "opinion_text", "content", "raw_text"):
                if candidate in record:
                    text_key = candidate
                    break
            if text_key is None:
                print("No recognizable text field found; aborting.")
                sys.exit(1)
            print(f"Using text field: '{text_key}'")
        records.append(record[text_key])
        if len(records) >= SAMPLE_N:
            break

    stats = [quality_flags(t) for t in records]
    chars = [s["chars"] for s in stats]
    printable = [s["printable_ratio"] for s in stats]
    markers = [s["legal_markers"] for s in stats]
    garbage = [s["garbage_token_ratio"] for s in stats]

    sample_chars = sum(chars)
    avg_mb_per_1k = sample_chars / len(stats) * 1000 / (1024 * 1024)
    records_for_100mb = TARGET_MB * (1024 * 1024) / (sample_chars / len(stats))
    chunks_for_100mb = int(TARGET_MB * (1024 * 1024) / CHUNK_SIZE)

    print("\n================ DATA SUITABILITY REPORT ================")
    print(f"Records sampled                 : {len(records)}")
    print(f"Text length  min/median/max     : {min(chars)} / {int(median(chars))} / {max(chars)} chars")
    print(f"Printable ratio  min/median     : {min(printable):.3f} / {median(printable):.3f}")
    print(f"Legal-marker hits min/median/max: {min(markers)} / {median(markers)} / {max(markers)}")
    print(f"Garbage-token ratio  median/max : {median(garbage):.4f} / {max(garbage):.4f}")
    print(f"Empty records                   : {sum(1 for s in stats if s.get('empty'))}")
    print(f"Records < 1000 chars (too short): {sum(1 for c in chars if c < 1000)}")
    print(f"Avg MB per 1,000 records        : {avg_mb_per_1k:.2f}")
    print(f"Records needed for {TARGET_MB} MB      : ~{int(records_for_100mb):,}")
    print(f"Chunks at size {CHUNK_SIZE}          : ~{chunks_for_100mb:,}")
    print("==========================================================\n")

    # Print two mid-length excerpts so a human can eyeball the text quality.
    order = sorted(range(len(records)), key=lambda i: chars[i])
    for label, idx in (("SHORT-median", order[len(order) // 2]), ("LONG", order[-1])):
        excerpt = " ".join(records[idx].split())[:600]
        print(f"--- {label} excerpt (record {idx}, {chars[idx]} chars) ---")
        print(excerpt + ("..." if chars[idx] > 600 else ""))
        print()


if __name__ == "__main__":
    main()