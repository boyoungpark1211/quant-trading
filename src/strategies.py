"""Strategy functions: take an OHLCV DataFrame, return a position series (1 = long, 0 = flat).

Long-only by design for the baseline — shorting crypto has different (and usually worse)
cost/liquidation dynamics that deserve separate treatment, not a flag on this function.
"""

import pandas as pd


def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Classic trend-following baseline: long while the fast SMA is above the slow SMA."""
    fast_sma = df["close"].rolling(fast).mean()
    slow_sma = df["close"].rolling(slow).mean()
    position = (fast_sma > slow_sma).astype(int)
    return position


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Benchmark: long from the first bar with valid data onward."""
    return pd.Series(1, index=df.index)
