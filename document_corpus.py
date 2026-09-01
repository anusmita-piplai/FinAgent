"""
document_corpus.py - Regulatory and Financial Disclosures Corpus & Semantic Retrieval Layer
PS-01: Multi-Agent Autonomous Financial Intelligence System for Retail Investors
"""

import sys
import os
import json
import math
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DocumentChunk:
    chunk_id: str
    ticker: str
    title: str
    doc_type: str  # e.g., "SEBI Filing", "Earnings Call Transcript", "Auditor Report", "Board Resolution"
    period: str    # e.g., "Q3 FY26", "Annual FY25"
    filing_date: str
    section: str
    content: str
    citation: str


# Rich synthetic & realistic financial disclosures corpus for NSE/BSE listed equities
DEFAULT_DOCUMENTS: List[Dict[str, Any]] = [
    # --- RELIANCE INDUSTRIES (RELIANCE.NS) ---
    {
        "ticker": "RELIANCE.NS",
        "title": "Reliance Industries Q3 FY26 Earnings Conference Call Transcript",
        "doc_type": "Earnings Call Transcript",
        "period": "Q3 FY26",
        "filing_date": "2026-01-20",
        "sections": [
            {
                "section": "Management Guidance & Capex",
                "content": "Jio Financial Services and Retail operations registered a 19.4% YoY EBITDA growth. Management affirmed that gross capital expenditure for FY26 will peak at ₹1,35,000 Crore with major allocations directed towards 5G advanced rollout, green hydrogen gigafactories in Jamnagar, and expansion of FMCG retail footprints. Net debt-to-equity ratio remains conservative at 0.38x."
            },
            {
                "section": "Oil-to-Chemicals (O2C) Performance",
                "content": "O2C segment margins improved with Singapore gross refining margins (GRM) averaging $9.8/bbl compared to $8.2/bbl in Q2. Cracker operating rates were sustained above 96%. Management expects domestic fuel demand to remain resilient amidst sustained infrastructure spending."
            },
            {
                "section": "Digital Services & Jio ARPU",
                "content": "Reliance Jio added 11.2 million net subscribers in Q3 FY26. Average Revenue Per User (ARPU) expanded from ₹181.7 to ₹195.4 following tariff rationalization in premium 5G unlimited plans. Subscriber churn decreased to 1.4% per month."
            }
        ]
    },
    {
        "ticker": "RELIANCE.NS",
        "title": "SEBI Clause 36 Disclosure - Green Energy Subsidies & Joint Venture",
        "doc_type": "SEBI Filing",
        "period": "Q3 FY26",
        "filing_date": "2026-01-12",
        "sections": [
            {
                "section": "Regulatory Disclosure - Material Event",
                "content": "Reliance New Energy Limited has signed a definitive 50:50 joint venture with a European electrolyser manufacturer for 2.5GW alkaline and PEM electrolyser manufacturing facility in Gujarat under the National Green Hydrogen Mission subsidy scheme (allocation of ₹1,480 Crore)."
            },
            {
                "section": "Risk Assessment & Regulatory Compliance",
                "content": "The company confirms full compliance with SEBI LODR Regulations 2015. No pending litigation or encumbrances exist against the new manufacturing assets. Environmental clearances obtained from the Ministry of Environment, Forest and Climate Change."
            }
        ]
    },
    {
        "ticker": "RELIANCE.NS",
        "title": "Reliance Industries Annual Risk & Governance Assessment",
        "doc_type": "Annual Report Excerpt",
        "period": "FY25-FY26",
        "filing_date": "2025-09-15",
        "sections": [
            {
                "section": "Risk Factors - Commodity & FX Sensitivity",
                "content": "Crude oil volatility and foreign currency fluctuations (USD/INR) remain key operational risks. A $1/bbl change in Brent crude prices historically impacts annual operating EBITDA by approximately ₹2,100 Crore. Hedging policy covers 60% of short-term open exposures."
            }
        ]
    },

    # --- TATA CONSULTANCY SERVICES (TCS.NS) ---
    {
        "ticker": "TCS.NS",
        "title": "Tata Consultancy Services Q3 FY26 Financial Results & Management Commentary",
        "doc_type": "Earnings Call Transcript",
        "period": "Q3 FY26",
        "filing_date": "2026-01-11",
        "sections": [
            {
                "section": "Deal TCV & Enterprise AI Adoption",
                "content": "Total Contract Value (TCV) for the quarter stood at $9.4 Billion, driven by BFSI cloud transformations and GenAI implementations. The AI Pipeline doubled sequentially to $1.8 Billion. Operating margin expanded 40 bps QoQ to 25.4%."
            },
            {
                "section": "Attrition & Headcount Guidance",
                "content": "LTM IT services attrition moderated further to 11.2%. Total headcount stood at 612,400 with net additions of 4,200 employees. Management reiterated target operating margin corridor of 26-28% for FY27."
            }
        ]
    },
    {
        "ticker": "TCS.NS",
        "title": "SEBI Compliance Filing - Share Buyback & Dividend Declaration",
        "doc_type": "SEBI Filing",
        "period": "Q3 FY26",
        "filing_date": "2026-01-11",
        "sections": [
            {
                "section": "Capital Return & Dividend",
                "content": "Board of Directors declared an interim dividend of ₹28 per equity share and approved special capital allocation policy maintaining >85% free cash flow payout to shareholders."
            }
        ]
    },

    # --- INFOSYS (INFY.NS) ---
    {
        "ticker": "INFY.NS",
        "title": "Infosys Limited Q3 FY26 Earnings Call & Guidance Revision",
        "doc_type": "Earnings Call Transcript",
        "period": "Q3 FY26",
        "filing_date": "2026-01-15",
        "sections": [
            {
                "section": "Revenue Guidance & Margin Walk",
                "content": "Infosys revised its FY26 constant currency revenue growth guidance upwards to 4.5%-5.2% (from 3.75%-4.5%). Large deal TCV came in at $3.2 Billion with 54% net new deals. Operating margin held steady at 21.2%."
            },
            {
                "section": "Macro Discretionary Spending",
                "content": "Discretionary spend in European retail and North American telecom continues to face elongated sales cycles, but BFSI and Manufacturing showed double-digit sequential recovery."
            }
        ]
    },

    # --- HDFC BANK (HDFCBANK.NS) ---
    {
        "ticker": "HDFCBANK.NS",
        "title": "HDFC Bank Q3 FY26 Investor Presentation & Regulatory Notes",
        "doc_type": "Earnings Call Transcript",
        "period": "Q3 FY26",
        "filing_date": "2026-01-18",
        "sections": [
            {
                "section": "Net Interest Margin (NIM) & Credit Growth",
                "content": "Core NIM expanded by 8 bps QoQ to 3.52%. Gross NPA declined to 1.24% with Net NPA at 0.31%. Credit deposit ratio (LDR) improved towards target threshold of 84.5% as deposit growth (18.2% YoY) outpaced credit growth (13.6% YoY)."
            },
            {
                "section": "Provision Coverage & Capital Adequacy",
                "content": "Provision Coverage Ratio (PCR) stood at 74.8%. Capital Adequacy Ratio (CRAR) under Basel III stood comfortably at 19.8% against regulatory minimum of 11.5%."
            }
        ]
    },

    # --- TATA MOTORS (TATAMOTORS.NS) ---
    {
        "ticker": "TATAMOTORS.NS",
        "title": "Tata Motors Demerger Disclosure & EV Profitability Note",
        "doc_type": "SEBI Filing",
        "period": "Q3 FY26",
        "filing_date": "2026-01-22",
        "sections": [
            {
                "section": "Corporate Demerger Progress",
                "content": "NCLT approval for demerger into two independent listed entities (Commercial Vehicles & Passenger/EV Vehicles) is in final stage. Operational separation completed across supply chains and technology stacks."
            },
            {
                "section": "EV & JLR Margin Outlook",
                "content": "JLR EBIT margin achieved 8.9% with order book standing at 148,000 units. Tata Passenger Electric Mobility (TPEM) reported positive EBITDA before battery cell localization incentives."
            }
        ]
    }
]


class SemanticSearchIndex:
    """
    Lightweight, robust semantic search index with TF-IDF vectorization,
    cosine similarity, and BM25 ranking for grounded retrieval.
    """

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None):
        self.chunks: List[DocumentChunk] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[int, float]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0

        raw_docs = documents or DEFAULT_DOCUMENTS
        self._build_corpus_chunks(raw_docs)
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # Remove punctuation, split into alphanumeric tokens
        tokens = re.findall(r'\b[a-z0-9_]{2,}\b', text)
        stopwords = {
            "the", "and", "is", "in", "it", "to", "for", "with", "on", "as", "at",
            "by", "an", "be", "this", "that", "from", "or", "are", "was", "were",
            "of", "has", "had", "have", "been", "will", "would", "shall", "can"
        }
        return [t for t in tokens if t not in stopwords]

    def _build_corpus_chunks(self, documents: List[Dict[str, Any]]) -> None:
        self.chunks.clear()
        chunk_idx = 1
        for doc in documents:
            ticker = doc.get("ticker", "UNKNOWN")
            title = doc.get("title", "Untitled Disclosure")
            doc_type = doc.get("doc_type", "Regulatory Disclosure")
            period = doc.get("period", "N/A")
            filing_date = doc.get("filing_date", "2026-01-01")

            for sec in doc.get("sections", []):
                sec_name = sec.get("section", "General")
                content = sec.get("content", "")
                citation = f"[{ticker} | {doc_type} | {period} | {sec_name} | Date: {filing_date}]"

                chunk = DocumentChunk(
                    chunk_id=f"CHK-{chunk_idx:04d}",
                    ticker=ticker,
                    title=title,
                    doc_type=doc_type,
                    period=period,
                    filing_date=filing_date,
                    section=sec_name,
                    content=content,
                    citation=citation
                )
                self.chunks.append(chunk)
                chunk_idx += 1

    def _build_index(self) -> None:
        """Calculate TF-IDF weights and BM25 index over all document chunks."""
        doc_count = len(self.chunks)
        if doc_count == 0:
            return

        doc_tokens_list: List[List[str]] = []
        df_counts: Dict[str, int] = {}
        total_tokens = 0

        for chunk in self.chunks:
            full_text = f"{chunk.title} {chunk.section} {chunk.content} {chunk.ticker} {chunk.doc_type}"
            tokens = self._tokenize(full_text)
            doc_tokens_list.append(tokens)
            self.doc_lengths.append(len(tokens))
            total_tokens += len(tokens)

            unique_tokens = set(tokens)
            for t in unique_tokens:
                df_counts[t] = df_counts.get(t, 0) + 1

        self.avg_doc_length = total_tokens / doc_count if doc_count > 0 else 1.0

        # Build vocabulary & IDF
        self.vocabulary = {t: idx for idx, t in enumerate(sorted(df_counts.keys()))}
        for token, df in df_counts.items():
            # Smoothed IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[token] = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))

        # Build sparse TF-IDF vectors
        self.doc_vectors = []
        for tokens in doc_tokens_list:
            tf_dict: Dict[str, int] = {}
            for t in tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1

            vec: Dict[int, float] = {}
            vec_sum_sq = 0.0
            for t, count in tf_dict.items():
                if t in self.vocabulary:
                    dim = self.vocabulary[t]
                    tfidf = (1.0 + math.log(count)) * self.idf.get(t, 1.0)
                    vec[dim] = tfidf
                    vec_sum_sq += tfidf * tfidf

            norm = math.sqrt(vec_sum_sq) if vec_sum_sq > 0 else 1.0
            normalized_vec = {dim: val / norm for dim, val in vec.items()}
            self.doc_vectors.append(normalized_vec)

    def search(
        self,
        query: str,
        ticker: Optional[str] = None,
        top_k: int = 3,
        min_score: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Search corpus for relevant chunks matching the query, optionally filtered by ticker.
        Returns ranked results with score, content, metadata, and citation attribution.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Vectorize query
        q_tf: Dict[str, int] = {}
        for t in query_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1

        q_vec: Dict[int, float] = {}
        q_sum_sq = 0.0
        for t, count in q_tf.items():
            if t in self.vocabulary:
                dim = self.vocabulary[t]
                weight = (1.0 + math.log(count)) * self.idf.get(t, 1.0)
                q_vec[dim] = weight
                q_sum_sq += weight * weight

        q_norm = math.sqrt(q_sum_sq) if q_sum_sq > 0 else 1.0
        q_vec = {dim: val / q_norm for dim, val in q_vec.items()}

        scores = []
        k1 = 1.2
        b = 0.75

        for idx, chunk in enumerate(self.chunks):
            if ticker and chunk.ticker.upper() != ticker.upper():
                continue

            # Cosine similarity
            doc_vec = self.doc_vectors[idx]
            cosine_sim = sum(val * doc_vec.get(dim, 0.0) for dim, val in q_vec.items())

            # BM25 boost
            doc_len = self.doc_lengths[idx]
            bm25_score = 0.0
            for t in query_tokens:
                if t in self.idf:
                    # Token freq in document
                    tf = sum(1 for tok in self._tokenize(chunk.content + " " + chunk.title) if tok == t)
                    numerator = tf * (k1 + 1.0)
                    denominator = tf + k1 * (1.0 - b + b * (doc_len / self.avg_doc_length))
                    bm25_score += self.idf[t] * (numerator / (denominator if denominator > 0 else 1.0))

            combined_score = round(0.6 * cosine_sim + 0.4 * min(bm25_score / 10.0, 1.0), 4)

            if combined_score >= min_score or len(scores) < top_k:
                scores.append({
                    "score": combined_score,
                    "chunk": chunk
                })

        # Rank by score descending
        scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = scores[:top_k]

        results = []
        for item in top_results:
            chunk_data = asdict(item["chunk"])
            chunk_data["relevance_score"] = item["score"]
            results.append(chunk_data)

        return results


# Global singleton instance for easy import across agent modules
_GLOBAL_INDEX = None

def get_corpus_index() -> SemanticSearchIndex:
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is None:
        _GLOBAL_INDEX = SemanticSearchIndex()
    return _GLOBAL_INDEX


def search_corpus(query: str, ticker: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
    """Public helper to search the financial disclosure corpus and return formatted results."""
    index = get_corpus_index()
    results = index.search(query=query, ticker=ticker, top_k=top_k)
    return {
        "status": "ok",
        "query": query,
        "ticker_filter": ticker,
        "count": len(results),
        "results": results
    }


if __name__ == "__main__":
    query_arg = sys.argv[1] if len(sys.argv) > 1 else "Capex plans, 5G rollout and debt profile"
    ticker_arg = sys.argv[2] if len(sys.argv) > 2 else "RELIANCE.NS"

    print(f"--- Searching Regulatory Corpus for: '{query_arg}' (Ticker: {ticker_arg}) ---")
    response = search_corpus(query=query_arg, ticker=ticker_arg, top_k=2)

    print(json.dumps(response, indent=2))
