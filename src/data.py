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

    rows = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit_per_call)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts == cursor:
            break
        cursor = last_ts + 1
        if len(batch) < limit_per_call:
            break

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache_path)
    return df
