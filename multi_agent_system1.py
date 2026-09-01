"""
multi_agent_system.py — Core multi-agent pipeline for FinAgent.

Defines:
    UserProfile               - dataclass for a retail investor persona
    DEFAULT_PROFILES          - 4 sample personas
    SignalClassifierAgent     - technical momentum/volume/RSI agent
    FundamentalRagAgent       - RAG-grounded fundamental sentiment agent
    RiskProfilerAgent         - persona-aware position sizing & suitability
    SynthesisAgent            - combines all agents into one verdict
    get_system_pipeline()     - returns a ready-to-use Pipeline instance
"""

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from market_data import fetch_market_data
from document_corpus import search_corpus


# ---------------------------------------------------------------------------
# USER PROFILE
# ---------------------------------------------------------------------------
@dataclass
class UserProfile:
    name: str
    risk_score: int                       # 1 (very conservative) - 10 (very aggressive)
    risk_category: str                    # display label
    total_capital: float                  # INR
    max_drawdown_tolerance_pct: float
    max_single_stock_allocation_pct: float
    current_holdings: List[str] = field(default_factory=list)


DEFAULT_PROFILES: Dict[str, UserProfile] = {
    "first_time_retail": UserProfile(
        name="Aarav — Gen-Z, ₹1L Starter Portfolio",
        risk_score=5,
        risk_category="Moderate (New Investor)",
        total_capital=100_000.0,
        max_drawdown_tolerance_pct=12.0,
        max_single_stock_allocation_pct=15.0,
        current_holdings=["NIFTYBEES", "INFY.NS"],
    ),
    "conservative_retiree": UserProfile(
        name="Sunita — Retiree, Capital Safety First",
        risk_score=2,
        risk_category="Conservative",
        total_capital=2_500_000.0,
        max_drawdown_tolerance_pct=6.0,
        max_single_stock_allocation_pct=8.0,
        current_holdings=["HDFCBANK.NS", "TCS.NS", "PPF"],
    ),
    "moderate_growth": UserProfile(
        name="Vikram — Balanced Mid-Career Investor",
        risk_score=6,
        risk_category="Moderate Growth",
        total_capital=800_000.0,
        max_drawdown_tolerance_pct=15.0,
        max_single_stock_allocation_pct=18.0,
        current_holdings=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
    ),
    "aggressive_trader": UserProfile(
        name="Pooja — Aggressive Momentum Trader",
        risk_score=9,
        risk_category="Aggressive",
        total_capital=400_000.0,
        max_drawdown_tolerance_pct=28.0,
        max_single_stock_allocation_pct=35.0,
        current_holdings=["TATAMOTORS.NS"],
    ),
}


def _latency_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


# ---------------------------------------------------------------------------
# AGENT 1: SIGNAL CLASSIFIER (technical: momentum + volume + RSI/EMA)
# ---------------------------------------------------------------------------
class SignalClassifierAgent:
    def analyze(self, market: Dict[str, Any], force_conflict: bool = False) -> Dict[str, Any]:
        start = time.perf_counter()

        prices = market["price_series"]
        volumes = market["volume_series"]

        # --- Momentum / trend ---
        pct_change = market["price_change_pct"]
        if abs(pct_change) < 0.5:
            trend_label = "Neutral / Consolidating"
        elif pct_change > 0:
            trend_label = "Bullish"
        else:
            trend_label = "Bearish"

        # --- RSI (Wilder-style, simplified) ---
        gains, losses = [], []
        for i in range(1, len(prices)):
            delta = prices[i] - prices[i - 1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        rsi = 100.0 if avg_loss == 0 else round(100 - (100 / (1 + (avg_gain / avg_loss))), 1)

        # --- Volume anomaly ---
        vol_verdict = "Normal"
        if len(volumes) >= 3:
            *baseline, latest = volumes
            avg_baseline = sum(baseline) / len(baseline) if baseline else latest
            ratio = latest / avg_baseline if avg_baseline else 1
            if ratio >= 2.0:
                vol_verdict = "Spike Detected"
            elif ratio <= 0.5:
                vol_verdict = "Drought"

        # Overall signal blends the three dimensions
        if rsi >= 70:
            signal = "Cautious — Overbought (RSI)"
        elif rsi <= 30:
            signal = "Bullish — Oversold Rebound Zone"
        elif trend_label == "Bullish" and vol_verdict == "Spike Detected":
            signal = "Strongly Bullish"
        elif trend_label == "Bearish":
            signal = "Bearish"
        else:
            signal = trend_label

        if force_conflict:
            # Force a strongly bullish technical read to set up a
            # deliberate contradiction with the fundamental agent.
            signal = "Strongly Bullish"

        confidence_pct = int(min(95, max(35, 50 + abs(pct_change) * 4 + (15 if vol_verdict == "Spike Detected" else 0))))

        cited_reasoning = [
            f"5-period price delta of {pct_change:+.2f}% classifies trend as {trend_label}.",
            f"RSI-14 at {rsi} is {'overbought' if rsi >= 70 else 'oversold' if rsi <= 30 else 'within neutral band'}.",
            f"Volume verdict: {vol_verdict}, based on latest vs. rolling baseline volume.",
        ]

        return {
            "signal": signal,
            "confidence_pct": confidence_pct,
            "metrics": {
                "rsi_14": rsi,
                "ema_trend": trend_label,
                "volume_verdict": vol_verdict,
            },
            "cited_reasoning": cited_reasoning,
            "execution_latency_ms": _latency_ms(start),
        }


# ---------------------------------------------------------------------------
# AGENT 2: FUNDAMENTAL RAG AGENT
# ---------------------------------------------------------------------------
class FundamentalRagAgent:
    _POSITIVE_WORDS = {
        "beat", "beats", "growth", "record", "upgrade", "upgraded", "strong",
        "profit", "surge", "rally", "outperform", "expansion", "raised",
        "improving", "stable", "confidence",
    }
    _NEGATIVE_WORDS = {
        "miss", "missed", "downgrade", "downgraded", "weak", "loss",
        "lawsuit", "probe", "investigation", "decline", "cut", "recall",
        "risk", "delay", "pressure", "scrutiny", "modest",
    }

    def analyze(self, ticker: str, force_conflict: bool = False) -> Dict[str, Any]:
        start = time.perf_counter()

        search_res = search_corpus(
            query="growth risk debt capex margin outlook",
            ticker=ticker,
            top_k=4,
        )
        chunks = search_res["results"]

        pos_hits, neg_hits = 0, 0
        positive_catalysts, key_risk_factors, citations = [], [], []

        for chunk in chunks:
            text_lower = chunk["content"].lower()
            words = set(text_lower.replace(",", " ").replace(".", " ").split())
            p = len(words & self._POSITIVE_WORDS)
            n = len(words & self._NEGATIVE_WORDS)
            pos_hits += p
            neg_hits += n
            citations.append(chunk["citation"])
            if p >= n:
                positive_catalysts.append(chunk["content"][:140])
            else:
                key_risk_factors.append(chunk["content"][:140])

        if not positive_catalysts:
            positive_catalysts = ["No strong positive catalysts identified in retrieved filings."]
        if not key_risk_factors:
            key_risk_factors = ["No material risk factors flagged in retrieved filings."]

        total = pos_hits + neg_hits
        if force_conflict:
            fundamental_stance = "Headwinds Flagged"
        elif total == 0:
            fundamental_stance = "Stable / Neutral"
        else:
            score = (pos_hits - neg_hits) / total
            if score > 0.15:
                fundamental_stance = "Growth Supportive"
            elif score < -0.15:
                fundamental_stance = "Headwinds Flagged"
            else:
                fundamental_stance = "Stable / Neutral"

        confidence_pct = int(min(90, max(40, 50 + abs(pos_hits - neg_hits) * 8)))

        return {
            "fundamental_stance": fundamental_stance,
            "confidence_pct": confidence_pct,
            "chunk_count": len(chunks),
            "positive_catalysts": positive_catalysts,
            "key_risk_factors": key_risk_factors,
            "retrieved_citations": citations or ["No citations retrieved"],
            "execution_latency_ms": _latency_ms(start),
        }


# ---------------------------------------------------------------------------
# AGENT 3: RETAIL RISK PROFILER
# ---------------------------------------------------------------------------
class RiskProfilerAgent:
    def analyze(self, ticker: str, current_price: float, profile: UserProfile,
                current_signal: str) -> Dict[str, Any]:
        start = time.perf_counter()

        # Suitability: how well this signal fits this investor's risk appetite
        signal_intensity = 1 if "Bullish" in current_signal else (-1 if "Bearish" in current_signal else 0)
        base_suitability = 50 + (signal_intensity * profile.risk_score * 4)
        suitability_score = int(max(5, min(98, base_suitability)))

        if suitability_score >= 75:
            suitability_verdict = "Strong Fit for Profile"
        elif suitability_score >= 50:
            suitability_verdict = "Moderate Fit — Proceed with Sizing Discipline"
        else:
            suitability_verdict = "Weak Fit — Consider Avoiding or Reducing Size"

        # Position sizing scaled by risk score (1-10) and allocation cap
        risk_fraction = profile.risk_score / 10
        target_alloc_pct = min(
            profile.max_single_stock_allocation_pct,
            profile.max_single_stock_allocation_pct * (0.4 + 0.6 * risk_fraction),
        )
        recommended_capital_inr = round(profile.total_capital * (target_alloc_pct / 100), 2)
        recommended_shares_qty = int(recommended_capital_inr // current_price) if current_price > 0 else 0

        suggested_stop_loss_pct = round(
            max(3.0, profile.max_drawdown_tolerance_pct * 0.4), 1
        )
        stop_loss_price_inr = round(current_price * (1 - suggested_stop_loss_pct / 100), 2)

        # Current concentration is a simulated existing-holdings weight
        current_weight_pct = round(
            min(profile.max_single_stock_allocation_pct * 0.6, 12.0), 1
        )

        behavioral_alerts = []
        if current_weight_pct + target_alloc_pct > profile.max_single_stock_allocation_pct:
            behavioral_alerts.append(
                f"Adding this position would exceed your {profile.max_single_stock_allocation_pct}% "
                "single-stock allocation cap — consider a smaller size."
            )
        if suitability_score < 50 and profile.risk_score <= 3:
            behavioral_alerts.append(
                "Signal direction conflicts with your conservative risk profile — "
                "flagged for extra caution before acting."
            )

        return {
            "user_risk_tier": profile.risk_category,
            "suitability_score": suitability_score,
            "suitability_verdict": suitability_verdict,
            "portfolio_metrics": {
                "recommended_capital_inr": recommended_capital_inr,
                "recommended_shares_qty": recommended_shares_qty,
                "stop_loss_price_inr": stop_loss_price_inr,
                "suggested_stop_loss_pct": suggested_stop_loss_pct,
                "current_weight_pct": current_weight_pct,
                "max_allowable_weight_pct": profile.max_single_stock_allocation_pct,
            },
            "behavioral_alerts": behavioral_alerts,
            "execution_latency_ms": _latency_ms(start),
        }


# ---------------------------------------------------------------------------
# AGENT 4: SYNTHESIS / CONSENSUS ORCHESTRATOR
# ---------------------------------------------------------------------------
class SynthesisAgent:
    def synthesize(self, ticker: str, market: Dict[str, Any], tech: Dict[str, Any],
                    rag: Dict[str, Any], risk: Dict[str, Any],
                    degraded_mode: bool = False) -> Dict[str, Any]:
        start = time.perf_counter()
        timestamp = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        tech_bullish = "Bullish" in tech["signal"] or "Strongly Bullish" in tech["signal"]
        tech_bearish = "Bearish" in tech["signal"]
        rag_positive = rag["fundamental_stance"] == "Growth Supportive"
        rag_negative = rag["fundamental_stance"] == "Headwinds Flagged"

        conflict_detected = (tech_bullish and rag_negative) or (tech_bearish and rag_positive)
        conflict_reason = ""
        if conflict_detected:
            conflict_reason = (
                f"Signal Screener reads '{tech['signal']}' while Fundamental RAG reads "
                f"'{rag['fundamental_stance']}' — resolved by weighting the higher-confidence "
                "agent and down-weighting the overall consensus confidence."
            )

        # Weighted directional score
        tech_dir = 1 if tech_bullish else (-1 if tech_bearish else 0)
        rag_dir = 1 if rag_positive else (-1 if rag_negative else 0)

        tech_w, rag_w, risk_w = 0.45, 0.35, 0.20
        if degraded_mode:
            # If data is degraded, lean more on whichever agent still has signal
            tech_w, rag_w = 0.3, 0.5

        weighted_score = (
            tech_dir * (tech["confidence_pct"] / 100) * tech_w
            + rag_dir * (rag["confidence_pct"] / 100) * rag_w
            + (1 if risk["suitability_score"] >= 60 else -1 if risk["suitability_score"] < 40 else 0)
              * (risk["suitability_score"] / 100) * risk_w
        )

        if weighted_score > 0.15:
            final_action = "BUY / ACCUMULATE"
        elif weighted_score < -0.15:
            final_action = "SELL / REDUCE"
        else:
            final_action = "NEUTRAL / HOLD"

        overall_confidence_pct = int(min(95, max(20, 55 + abs(weighted_score) * 100)))
        if conflict_detected:
            overall_confidence_pct = int(overall_confidence_pct * 0.75)
        if degraded_mode:
            overall_confidence_pct = int(overall_confidence_pct * 0.8)

        verdict_summary = (
            f"Consensus verdict for {ticker}: {final_action}, driven primarily by "
            f"{'the Signal Screener' if abs(tech_dir * tech_w) >= abs(rag_dir * rag_w) else 'the Fundamental RAG agent'}. "
            f"Suitability for the active profile is scored {risk['suitability_score']}/100 "
            f"({risk['suitability_verdict']})."
        )
        if degraded_mode:
            verdict_summary += " Note: market feed is running in degraded mode; confidence has been damped accordingly."

        reasoning_trace = [
            {
                "step": 1,
                "agent": "Signal Screener",
                "finding": f"Technical read: {tech['signal']} (confidence {tech['confidence_pct']}%).",
                "citation": "RSI-14 / EMA / Volume computation over recent price-volume series",
                "timestamp": timestamp(),
            },
            {
                "step": 2,
                "agent": "Fundamental RAG",
                "finding": f"Fundamental stance: {rag['fundamental_stance']} (confidence {rag['confidence_pct']}%), grounded in {rag['chunk_count']} retrieved filing chunk(s).",
                "citation": rag["retrieved_citations"][0] if rag["retrieved_citations"] else "No citation",
                "timestamp": timestamp(),
            },
            {
                "step": 3,
                "agent": "Retail Risk Profiler",
                "finding": f"Suitability {risk['suitability_score']}/100 for active persona; recommended sizing ₹{risk['portfolio_metrics']['recommended_capital_inr']:,.0f}.",
                "citation": "Persona risk parameters × signal direction × allocation caps",
                "timestamp": timestamp(),
            },
        ]
        if conflict_detected:
            reasoning_trace.append({
                "step": 4,
                "agent": "Synthesis Orchestrator",
                "finding": f"Conflict detected between technical and fundamental agents. {conflict_reason}",
                "citation": "Cross-agent consensus reconciliation logic",
                "timestamp": timestamp(),
            })
        reasoning_trace.append({
            "step": len(reasoning_trace) + 1,
            "agent": "Synthesis Orchestrator",
            "finding": f"Final weighted verdict: {final_action} at {overall_confidence_pct}% consensus confidence.",
            "citation": "Weighted multi-agent score aggregation",
            "timestamp": timestamp(),
        })

        return {
            "final_action": final_action,
            "overall_confidence_pct": overall_confidence_pct,
            "verdict_summary": verdict_summary,
            "conflict_detected": conflict_detected,
            "conflict_reason": conflict_reason,
            "reasoning_trace": reasoning_trace,
            "execution_latency_ms": _latency_ms(start),
        }


# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self):
        self.signal_agent = SignalClassifierAgent()
        self.rag_agent = FundamentalRagAgent()
        self.risk_agent = RiskProfilerAgent()
        self.synthesis_agent = SynthesisAgent()

    def run_full_pipeline(self, ticker: str, user_profile: UserProfile,
                           degraded_mode: bool = False,
                           force_conflict: bool = False) -> Dict[str, Any]:
        market = fetch_market_data(ticker, degraded=degraded_mode)

        tech_res = self.signal_agent.analyze(market, force_conflict=force_conflict)
        rag_res = self.rag_agent.analyze(ticker, force_conflict=force_conflict)
        risk_res = self.risk_agent.analyze(
            ticker, market["latest_close"], user_profile, tech_res["signal"]
        )
        synth_res = self.synthesis_agent.synthesize(
            ticker, market, tech_res, rag_res, risk_res, degraded_mode=degraded_mode
        )

        return {
            "market_data": market,
            "technical_analysis": tech_res,
            "fundamental_rag": rag_res,
            "risk_profiling": risk_res,
            "synthesis": synth_res,
        }


_pipeline_singleton: Optional[Pipeline] = None


def get_system_pipeline() -> Pipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = Pipeline()
    return _pipeline_singleton


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = get_system_pipeline()
    profile = DEFAULT_PROFILES["aggressive_trader"]
    result = pipeline.run_full_pipeline("RELIANCE.NS", profile, degraded_mode=False, force_conflict=False)
    import json
    print(json.dumps(result, indent=2, default=str))
