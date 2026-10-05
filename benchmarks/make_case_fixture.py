#!/usr/bin/env python3
"""Generate the 15-page case-file fixture PDF for upload testing.

Run from the repo root:
    python benchmarks\\make_case_fixture.py

Recreates benchmarks/fixtures/kestrel_case_file.pdf - an information-dense
litigation case file (cover sheet, engagement letter, fact chronology,
contract exhibits, correspondence, docket, damages model) with ~60
distinct retrievable facts. Authored content, no AI, fully deterministic.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kestrel_case_file.pdf"

# 15 pages of dense authored content. Facts are deliberately spread so
# retrieval has to find the right page, and include precision traps
# (similar names, gross-vs-net damages, lookalike dates).
PAGES = [
    # p1 - cover sheet
    [
        "SUPERIOR COURT OF THE STATE OF DELAWARE",
        "IN AND FOR NEW CASTLE COUNTY",
        "",
        "HORIZON LOGISTICS LLC,",
        "                    Plaintiff,",
        "v.                                        C.A. No. N24C-03-118",
        "STERLING FREIGHT SYSTEMS INC.,",
        "                    Defendant.",
        "",
        "CASE COVER SHEET",
        "",
        "Filed: 14 March 2026            Trial date: 9 November 2026",
        "Presiding judge: Hon. Marisol Vega-Katz",
        "Case type: breach of contract; unjust enrichment",
        "Plaintiff: Horizon Logistics LLC (Delaware LLC, Wilmington)",
        "Defendant: Sterling Freight Systems Inc. (Dover, Delaware)",
        "Trial preference: bench trial (no jury demand filed)",
    ],
    # p2 - engagement letter
    [
        "ENGAGEMENT LETTER - RIVERDALE & CHO LLP (EXTRACT 1/2)",
        "",
        "Client: Horizon Logistics LLC        Effective: 5 January 2026",
        "Matter: Horizon Logistics LLC v. Sterling Freight Systems Inc.",
        "",
        "Lead counsel: Dana Riverdale (admitted Delaware, 2009) - $650/hr",
        "Associate: Priya Cho - $425/hr",
        "Paralegal support: $175/hr",
        "",
        "A retainer of $75,000 is held in the client trust account and",
        "reconciled against monthly invoices. Invoices are issued monthly,",
        "net-30. Conflict check against Sterling Freight Systems cleared",
        "2 January 2026 with no conflicts identified.",
    ],
    # p3 - engagement terms cont.
    [
        "ENGAGEMENT LETTER - RIVERDALE & CHO LLP (EXTRACT 2/2)",
        "",
        "Pre-trial phase fee cap: $250,000 (excludes e-discovery and",
        "expert fees).",
        "",
        "E-discovery vendor: LexFetch Solutions - estimated $18,000.",
        "Deposition budget: $32,000 covering up to nine (9) depositions.",
        "",
        "Mediation is scheduled for 12 June 2026 in Wilmington with",
        "retired judge Alan Pettiway. If mediation fails, the parties",
        "will proceed to non-binding arbitration administered by JAMS",
        "with a single arbitrator; arbitration fees are split 50/50.",
    ],
    # p4 - background facts
    [
        "STATEMENT OF FACTS (EXTRACT 1/3) - BACKGROUND",
        "",
        "On 12 January 2024 the parties executed a Master Freight",
        "Services Agreement (the MSA) for a thirty-six (36) month term.",
        "Horizon Logistics provides warehousing and inventory control;",
        "Sterling Freight provides line-haul freight operations.",
        "",
        "The MSA sets a monthly minimum billing of $210,000. Invoices",
        "may be disputed within a fifteen (15) day window. The MSA is",
        "governed by Delaware law. Operations began 1 February 2024",
        "after a four-week onboarding audit.",
    ],
    # p5 - dispute chronology
    [
        "STATEMENT OF FACTS (EXTRACT 2/3) - DISPUTE CHRONOLOGY",
        "",
        "3 February 2025 - Sterling misses the delivery SLA for the",
        "first time; eleven (11) late shipments recorded in February.",
        "14 March 2025 - Horizon sends a cure notice by email",
        "(L. Okafor to R. Feldman); the MSA cure period is 30 days.",
        "18 April 2025 - second cure notice sent by certified mail.",
        "22 May 2025 - Sterling issues a credit of $48,750 covering",
        "twenty-three (23) failed lanes.",
        "30 June 2025 - Horizon withholds $120,400 from payment,",
        "asserting setoff for SLA credits.",
        "15 July 2025 - Sterling suspends services.",
        "2 August 2025 - Horizon engages Crestline Transport as",
        "replacement carrier at a $9,200 per week premium.",
    ],
    # p6 - damages narrative
    [
        "STATEMENT OF FACTS (EXTRACT 3/3) - DAMAGES NARRATIVE",
        "",
        "Replacement premium paid to Crestline Transport from 2 August",
        "to 30 November 2025 (seventeen weeks): $156,400 total.",
        "",
        "Lost contract revenue claimed by Horizon per the CFO audit",
        "memo dated 5 January 2026: $410,000.",
        "Storage overflow costs across twenty-three (23) incidents:",
        "$61,250. Customer refunds paid to fourteen (14) downstream",
        "customers: $88,900.",
        "",
        "Gross damages claimed: $716,550, plus pre-judgment interest",
        "at 5 percent per annum accruing from 15 July 2025.",
    ],
    # p7 - MSA clauses 1
    [
        "EXHIBIT A - MASTER FREIGHT SERVICES AGREEMENT (CLAUSES 1/2)",
        "",
        "Section 7.2 - Termination for convenience: either party may",
        "terminate on ninety (90) days written notice, subject to an",
        "early termination fee of $60,000.",
        "Section 7.3 - Termination for cause: thirty (30) day cure",
        "period following written notice of material breach.",
        "",
        "Section 9.1 - Limitation of liability: the greater of six",
        "(6) months of fees or $500,000, excepting gross negligence",
        "and willful misconduct.",
        "Section 9.4 - Indemnification is mutual and capped per 9.1.",
        "Section 5.6 - Late payment interest: 1.5 percent per month.",
        "Section 11.3 - Audit rights: twice per year on ten (10)",
        "business days notice.",
    ],
    # p8 - MSA clauses 2
    [
        "EXHIBIT A - MASTER FREIGHT SERVICES AGREEMENT (CLAUSES 2/2)",
        "",
        "Section 3.1 - On-time delivery: at least ninety-six percent",
        "(96%) of shipments within the 48-hour delivery window each",
        "calendar month.",
        "Section 3.4 - SLA credits: two percent (2%) of monthly",
        "billing per missed lane-week, capped at ten percent (10%)",
        "of monthly billing in any month.",
        "",
        "Section 4.2 - Force majeure excludes labor disputes of the",
        "affected party's own workforce.",
        "Section 6.1 - Insurance: $5,000,000 commercial general",
        "liability per party, certificates exchanged annually.",
        "Section 12.4 - Amendments require writing signed by both",
        "chief financial officers.",
    ],
    # p9 - NDA exhibit
    [
        "EXHIBIT B - MUTUAL NON-DISCLOSURE AGREEMENT (SIGNED 9 JAN 2024)",
        "",
        "Parties: Horizon Logistics LLC and Sterling Freight Systems",
        "Inc. Executed 9 January 2024, exchange of operational data",
        "for MSA diligence.",
        "",
        "Confidential period: five (5) years from disclosure.",
        "Return or destruction of confidential materials within",
        "fifteen (15) days of written request.",
        "Residuals: expressly excluded - no residuals rights are",
        "granted to either party.",
        "Governing law: New York.",
        "Liquidated damages for breach: $250,000 per violation.",
    ],
    # p10 - correspondence 1
    [
        "EXHIBIT C - CORRESPONDENCE LOG (1/2)",
        "",
        "14 March 2025 - Email, L. Okafor (Horizon VP Operations) to",
        "R. Feldman (Sterling COO): cure within 30 days or Horizon",
        "will exercise termination rights under Section 7.3.",
        "2 April 2025 - Reply, R. Feldman: attributes delays to",
        "winter weather; requests tolling of the cure period.",
        "18 April 2025 - Certified letter: formal second cure notice.",
        "22 July 2025 - Letter, Sterling to Horizon: counter-threat",
        "of termination and demand for $95,000 in unpaid invoices.",
        "30 July 2025 - Reply, Horizon: acknowledges the $95,000",
        "invoice balance but asserts setoff of $120,400 in SLA",
        "credits and overbilling.",
    ],
    # p11 - correspondence 2
    [
        "EXHIBIT C - CORRESPONDENCE LOG (2/2)",
        "",
        "8 September 2025 - Demand letter from Whitmore & Slocum",
        "(Sterling's counsel) threatening suit for $215,000.",
        "19 September 2025 - Response by Riverdale & Cho: rejects",
        "the demand and asserts accrued SLA credits of $73,500.",
        "24 October 2025 - Tolling agreement: the statute of",
        "limitations is tolled through 31 March 2026.",
        "6 November 2025 - Litigation hold notice issued to",
        "Sterling covering dispatch logs and lane-level GPS data.",
    ],
    # p12 - court filings 1
    [
        "DOCKET SUMMARY (1/2)",
        "",
        "14 March 2026 - Complaint filed (Dkt. 1): three counts -",
        "(I) breach of the MSA; (II) breach of the implied covenant",
        "of good faith; (III) unjust enrichment.",
        "21 March 2026 - Summons served on Sterling's registered",
        "agent, Corporation Trust Company.",
        "9 April 2026 - Answer and affirmative defenses (Dkt. 8):",
        "five (5) defenses including force majeure and waiver.",
        "23 April 2026 - Reply to affirmative defenses (Dkt. 10).",
        "30 April 2026 - Scheduling order (Dkt. 11): fact discovery",
        "closes 31 July 2026; expert reports 28 August 2026;",
        "dispositive motions due 18 September 2026.",
        "12 May 2026 - Motion to compel dispatch logs filed by",
        "Horizon (Dkt. 14).",
    ],
    # p13 - court filings 2
    [
        "DOCKET SUMMARY (2/2)",
        "",
        "29 May 2026 - Order (Dkt. 19): motion to compel granted in",
        "part; Sterling to produce dispatch logs by 12 June 2026;",
        "sanctions of $7,500 in fees awarded to Horizon.",
        "16 June 2026 - Sterling moves for a protective order",
        "(Dkt. 22) covering GPS telemetry.",
        "3 July 2026 - Ruling: protective order denied as premature.",
        "",
        "Depositions noticed: R. Feldman (Sterling COO) on",
        "21 July 2026; L. Okafor (Horizon VP Operations) on",
        "23 July 2026; S. Iyer (Horizon CFO) on 28 July 2026.",
    ],
    # p14 - damages model
    [
        "EXHIBIT D - DAMAGES MODEL (EXTRACT 1/2)",
        "",
        "Prepared by Dr. Elena Marquez, forensic accountant.",
        "",
        "Category                        Amount",
        "Lost contract revenue           $410,000",
        "Replacement carrier premium     $156,400",
        "Storage overflow costs          $61,250",
        "Customer refunds                $88,900",
        "Gross damages                   $716,550",
        "Less: mitigation credit        -$48,750",
        "NET DAMAGES CLAIMED             $667,800",
        "",
        "Model: but-for scenario over a 24-month lookback; pre-judgment",
        "interest at 5 percent per annum from 15 July 2025.",
    ],
    # p15 - expert + witnesses
    [
        "EXHIBIT D - DAMAGES MODEL (EXTRACT 2/2) + WITNESS LIST",
        "",
        "Expert engagement: Dr. Elena Marquez, forensic accountant,",
        "fee $450 per hour, capped at 120 hours ($54,000 maximum).",
        "",
        "Witness list (8): R. Feldman (Sterling COO); L. Okafor",
        "(Horizon VP Operations); S. Iyer (Horizon CFO); T. Brandt",
        "(Sterling CFO); M. Ruiz (Sterling dispatch supervisor);",
        "J. Whitaker (Crestline operations manager); Dr. Elena",
        "Marquez (damages expert); records custodian for dispatch",
        "logs.",
        "Exhibits A-1 through A-14; designation deadline 2 October",
        "2026. Designations protected as attorney work product.",
    ],
]


def make() -> Path:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(FIXTURE), pagesize=letter)
    for lines in PAGES:
        y = 730
        for ln in lines:
            c.drawString(72, y, ln)
            y -= 20
        c.showPage()
    c.save()
    return FIXTURE


if __name__ == "__main__":
    print(f"Wrote {make()} ({len(PAGES)} pages)")