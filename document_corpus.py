"""
document_corpus.py — Synthetic regulatory/financial document corpus with a
simple TF-style keyword search, standing in for a real vector DB + SEBI
filing ingestion pipeline for the hackathon sprint.
"""

import re
from collections import Counter

DEFAULT_DOCUMENTS = [
    {
        "id": "doc_001",
        "ticker": "RELIANCE.NS",
        "title": "Reliance Industries Q1 FY26 Earnings Call Transcript",
        "doc_type": "Earnings Transcript",
        "period": "Q1 FY26",
        "section": "Management Commentary — Retail & Jio Segment",
        "content": (
            "Management highlighted double-digit EBITDA growth in the retail "
            "and digital services segments, alongside continued capex "
            "discipline. Debt levels remain within the guided range, though "
            "the team flagged near-term margin pressure from crude volatility."
        ),
        "citation": "RIL Q1 FY26 Earnings Call, Slide 12",
    },
    {
        "id": "doc_002",
        "ticker": "RELIANCE.NS",
        "title": "Reliance Industries SEBI LODR Disclosure",
        "doc_type": "SEBI Filing",
        "period": "FY26",
        "section": "Related Party Transactions",
        "content": (
            "The company disclosed a related-party transaction under review "
            "by the audit committee; no adverse findings have been reported "
            "to date. Risk factors include ongoing regulatory scrutiny of "
            "certain group-level transactions."
        ),
        "citation": "SEBI LODR Filing, Regulation 23, FY26",
    },
    {
        "id": "doc_003",
        "ticker": "TCS.NS",
        "title": "TCS Q1 FY26 Earnings Call Transcript",
        "doc_type": "Earnings Transcript",
        "period": "Q1 FY26",
        "section": "Deal Pipeline & Attrition",
        "content": (
            "Deal wins were modest this quarter versus guidance, with "
            "management citing delayed client decision-making in North "
            "America. Attrition improved sequentially. Growth outlook for "
            "the back half of the year is described as cautiously stable."
        ),
        "citation": "TCS Q1 FY26 Earnings Call, Analyst Q&A",
    },
    {
        "id": "doc_004",
        "ticker": "TCS.NS",
        "title": "TCS Auditor's Report Note",
        "doc_type": "Auditor Note",
        "period": "FY25 Annual Report",
        "section": "Key Audit Matters",
        "content": (
            "No material weaknesses identified. Auditors note normal-course "
            "provisioning for doubtful receivables consistent with prior "
            "years; no going-concern qualifications raised."
        ),
        "citation": "TCS FY25 Annual Report, Auditor's Report, p.142",
    },
    {
        "id": "doc_005",
        "ticker": "INFY.NS",
        "title": "Infosys Q1 FY26 Earnings Call Transcript",
        "doc_type": "Earnings Transcript",
        "period": "Q1 FY26",
        "section": "Revenue Growth & Guidance",
        "content": (
            "Revenue growth beat street estimates on strong large-deal "
            "conversion. Management raised full-year revenue growth "
            "guidance and highlighted improved capex discipline across "
            "delivery centers."
        ),
        "citation": "Infosys Q1 FY26 Earnings Call, CFO Remarks",
    },
    {
        "id": "doc_006",
        "ticker": "INFY.NS",
        "title": "Infosys SEBI Filing — Board Resolution",
        "doc_type": "SEBI Filing",
        "period": "FY26",
        "section": "Capital Allocation",
        "content": (
            "Board approved a buyback program as part of ongoing capital "
            "allocation strategy. No pending litigation of material "
            "financial impact was disclosed for the period."
        ),
        "citation": "SEBI Filing, Board Resolution, FY26",
    },
    {
        "id": "doc_007",
        "ticker": "HDFCBANK.NS",
        "title": "HDFC Bank Q1 FY26 Earnings Call Transcript",
        "doc_type": "Earnings Transcript",
        "period": "Q1 FY26",
        "section": "Asset Quality & NIM",
        "content": (
            "Net interest margins held stable quarter-on-quarter. "
            "Management reported improving asset quality with a modest "
            "decline in gross NPAs, and reiterated confidence in loan "
            "growth for the remainder of the fiscal year."
        ),
        "citation": "HDFC Bank Q1 FY26 Earnings Call, CFO Commentary",
    },
    {
        "id": "doc_008",
        "ticker": "TATAMOTORS.NS",
        "title": "Tata Motors SEBI Disclosure — JLR Segment",
        "doc_type": "SEBI Filing",
        "period": "FY26",
        "section": "Segment Risk Factors",
        "content": (
            "The company disclosed continued supply-chain risk in the JLR "
            "segment related to semiconductor availability, alongside "
            "delay risk on certain new model launches. EV segment capex "
            "remains elevated relative to guidance."
        ),
        "citation": "SEBI LODR Filing, Segment Disclosures, FY26",
    },
]


def get_corpus_index() -> dict:
    """Returns a simple {ticker: [doc,...]} index over DEFAULT_DOCUMENTS."""
    index = {}
    for doc in DEFAULT_DOCUMENTS:
        index.setdefault(doc["ticker"], []).append(doc)
    return index


def _tokenize(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def search_corpus(query: str, ticker: str = None, top_k: int = 4) -> dict:
    """
    Very small BM25-style keyword search over DEFAULT_DOCUMENTS.
    Returns {"count": int, "results": [doc_with_relevance_score, ...]}
    sorted by relevance, optionally filtered to a single ticker.
    """
    query_terms = _tokenize(query)
    candidates = DEFAULT_DOCUMENTS if not ticker else [
        d for d in DEFAULT_DOCUMENTS if d["ticker"] == ticker
    ]

    scored = []
    for doc in candidates:
        doc_terms = _tokenize(doc["content"] + " " + doc["title"] + " " + doc["section"])
        overlap = sum(min(query_terms[t], doc_terms[t]) for t in query_terms)
        doc_len = sum(doc_terms.values()) or 1
        # crude BM25-ish score: term overlap normalized by doc length
        score = round(overlap / (doc_len ** 0.3), 3) if overlap else 0.0
        if score > 0:
            scored.append({**doc, "relevance_score": score})

    scored.sort(key=lambda d: d["relevance_score"], reverse=True)

    if not scored:
        # fall back to ticker-filtered (or all) docs at a low base score so
        # the RAG agent always has something to ground on
        fallback = candidates[:top_k]
        scored = [{**d, "relevance_score": 0.15} for d in fallback]

    return {"count": len(scored[:top_k]), "results": scored[:top_k]}

