"""
market_data.py — Simulated (near-real-time-style) NSE market data feed.

For a hackathon sprint, we generate deterministic-but-varied synthetic
price/volume data per ticker instead of hitting a paid live feed. Swap
`fetch_market_data` internals for a real API (yfinance, NSE API, etc.)
later without changing the return contract.
"""

import random
import zlib
from datetime import datetime, timezone


def _stable_seed(text: str) -> int:
    """Deterministic seed across process runs (Python's hash() is randomized per-process)."""
    return zlib.crc32(text.encode("utf-8"))

_BASE_PRICES = {
    "RELIANCE.NS": 2850.0,
    "TCS.NS": 3960.0,
    "INFY.NS": 1585.0,
    "HDFCBANK.NS": 1642.0,
    "TATAMOTORS.NS": 980.0,
}


def fetch_market_data(ticker: str, degraded: bool = False) -> dict:
    """
    Returns a snapshot of market data for `ticker`.

    If degraded=True, simulates a partially unavailable feed (used for
    the "degraded-data scenario" requirement) by returning a flagged,
    lower-confidence snapshot instead of raising.
    """
    rng = random.Random(_stable_seed(ticker.upper()) % 10_000)
    base = _BASE_PRICES.get(ticker.upper(), 1000.0)

    if degraded:
        return {
            "ticker": ticker,
            "latest_close": round(base, 2),
            "price_change_pct": 0.0,
            "volume_change_pct": 0.0,
            "price_series": [base] * 10,
            "volume_series": [],
            "feed_status": "DEGRADED",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    price_series = [base]
    for _ in range(9):
        price_series.append(round(price_series[-1] * (1 + rng.uniform(-0.015, 0.02)), 2))

    volume_series = [rng.randint(1_500_000, 8_500_000) for _ in range(6)]
    # Occasionally inject a volume spike so the volume agent has something to catch
    if rng.random() > 0.5:
        volume_series[-1] = int(volume_series[-1] * rng.uniform(2.0, 3.5))

    price_change_pct = round((price_series[-1] - price_series[0]) / price_series[0] * 100, 2)
    volume_change_pct = round(
        (volume_series[-1] - sum(volume_series[:-1]) / len(volume_series[:-1]))
        / (sum(volume_series[:-1]) / len(volume_series[:-1])) * 100, 2
    )

    return {
        "ticker": ticker,
        "latest_close": price_series[-1],
        "price_change_pct": price_change_pct,
        "volume_change_pct": volume_change_pct,
        "price_series": price_series,
        "volume_series": volume_series,
        "feed_status": "LIVE",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
