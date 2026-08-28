"""OHLCV fetching via ccxt, with a local CSV cache so repeated backtest runs don't re-hit the exchange."""

from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1d",
    since: str = "2018-01-01",
    limit_per_call: int = 1000,
) -> pd.DataFrame:
    """Fetch full OHLCV history for `symbol` from `exchange_id`'s public API, paginating as needed.

    `since` is an ISO date string. Uses only public market-data endpoints — no API key required.
    """
    cache_path = DATA_DIR / f"{exchange_id}_{symbol.replace('/', '-')}_{timeframe}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col="timestamp", parse_dates=True)

    exchange = getattr(ccxt, exchange_id)()
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")
    now_ms = exchange.milliseconds()

    rows = []
    cursor = since_ms
    while cursor < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit_per_call)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            # exchange returned a page that didn't move the cursor forward (e.g. hit a cap
            # smaller than limit_per_call and kept re-serving the same window) — bail out
            # rather than looping forever.
            break
        cursor = last_ts + 1

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache_path)
    return df


def fetch_daily_funding_rate(symbol: str = "BTC/USDT", since: str = "2019-09-01") -> pd.Series:
    """Daily-aggregated historical funding rate for a USDM perpetual (paid/received every 8h on Binance).

    Returns a Series indexed by day (midnight UTC) of that day's total funding rate — the real cost
    (or income) of holding a long position for the day, not an assumption.
    """
    cache_path = DATA_DIR / f"funding_{symbol.replace('/', '-')}.csv"
    if cache_path.exists():
        s = pd.read_csv(cache_path, index_col="timestamp", parse_dates=True)["funding_rate"]
        return s

    exchange = ccxt.binanceusdm()
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")

    rows = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        if last_ts == cursor:
            break
        cursor = last_ts + 1
        if len(batch) < 1000:
            break

    raw = pd.DataFrame(rows)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms")
    raw = raw.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    daily = raw["fundingRate"].resample("1D").sum()

    DATA_DIR.mkdir(exist_ok=True)
    daily.rename("funding_rate").to_csv(cache_path, index_label="timestamp")
    return daily
