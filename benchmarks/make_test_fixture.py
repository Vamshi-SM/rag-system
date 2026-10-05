#!/usr/bin/env python3
"""Generate the deterministic test fixture PDF for the upload test suite.

Run from the repo root:
    python benchmarks\\make_test_fixture.py

Recreates benchmarks/fixtures/kestrel_msa.pdf - a fictional 3-page MSA
whose facts are spread across pages so retrieval (local and RAG Engine)
has to find the right page, not just the top of the document.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kestrel_msa.pdf"

PAGES = [
    [
        "KESTREL PATH MASTER SERVICES AGREEMENT (EXTRACT 1/3)",
        "",
        "The monthly retainer cap under the Kestrel Path agreement is 2,400",
        "credits, invoiced at the start of each service month and reconciled",
        "quarterly against hours actually delivered.",
        "",
        "All work product delivered under this agreement remains the",
        "intellectual property of Kestrel Path until final acceptance",
        "is confirmed in writing by the client.",
    ],
    [
        "KESTREL PATH MASTER SERVICES AGREEMENT (EXTRACT 2/3)",
        "",
        "Client may terminate for convenience upon sixty (60) days written",
        "notice to Kestrel Path, paying a termination fee of 5,000 credits",
        "within thirty (30) days of the effective termination date.",
        "",
        "Neither party may assign this agreement without the prior written",
        "consent of the other party, except in connection with a merger.",
    ],
    [
        "KESTREL PATH MASTER SERVICES AGREEMENT (EXTRACT 3/3)",
        "",
        "Billable overtime is billed at 1.5 times the base rate, capped at",
        "twelve (12) hours per week without prior written approval.",
        "",
        "Engagement data is retained for ninety (90) days after the",
        "engagement ends, after which it is securely destroyed.",
        "",
        "This agreement is governed by the laws of the State of Delaware.",
    ],
]


def make() -> Path:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(FIXTURE), pagesize=letter)
    for lines in PAGES:
        y = 720
        for ln in lines:
            c.drawString(72, y, ln)
            y -= 22
        c.showPage()
    c.save()
    return FIXTURE


if __name__ == "__main__":
    print(f"Wrote {make()}")