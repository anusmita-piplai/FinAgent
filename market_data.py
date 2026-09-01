import sys
from datetime import datetime, timezone
import yfinance as yf


def fetch_market_data(ticker_symbol: str = "RELIANCE.NS") -> dict:
    """Fetch latest market data for a given ticker symbol using yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch recent daily history (5 days to ensure enough data points for change calculations)
        history = ticker.history(period="5d")

        if history.empty or len(history) < 1:
            return {
                "ticker": ticker_symbol,
                "status": "error",
                "message": f"No data found for ticker '{ticker_symbol}'",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        latest = history.iloc[-1]
        latest_close = round(float(latest["Close"]), 2)
        latest_volume = int(latest["Volume"])

        if len(history) >= 2:
            prev = history.iloc[-2]
            prev_close = float(prev["Close"])
            prev_volume = float(prev["Volume"])

            price_change_pct = round(((latest_close - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
            volume_change_pct = round(((latest_volume - prev_volume) / prev_volume) * 100, 2) if prev_volume else 0.0
        else:
            price_change_pct = 0.0
            volume_change_pct = 0.0

        # Use the date/time of the latest candle or current UTC time
        latest_ts = latest.name.isoformat() if hasattr(latest.name, "isoformat") else datetime.now(timezone.utc).isoformat()

        return {
            "ticker": ticker_symbol,
            "status": "ok",
            "latest_close": latest_close,
            "price_change_pct": price_change_pct,
            "latest_volume": latest_volume,
            "volume_change_pct": volume_change_pct,
            "timestamp": latest_ts,
        }
    except Exception as e:
        return {
            "ticker": ticker_symbol,
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    result = fetch_market_data(ticker_arg)
    print(result)
