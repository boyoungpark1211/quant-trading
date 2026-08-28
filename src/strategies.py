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


def donchian_breakout(df: pd.DataFrame, entry_lookback: int = 20, exit_lookback: int = 10) -> pd.Series:
    """Momentum/breakout: go long on a new `entry_lookback`-day high, exit on a new
    `exit_lookback`-day low. Classic dual-channel trend system (a simplified "turtle" rule) —
    a different hypothesis than SMA crossover: react to price extremes, not moving-average order.
    """
    entry_signal = df["close"] >= df["close"].rolling(entry_lookback).max().shift(1)
    exit_signal = df["close"] <= df["close"].rolling(exit_lookback).min().shift(1)

    position = pd.Series(0, index=df.index)
    in_position = False
    for i in range(len(df)):
        if not in_position and entry_signal.iloc[i]:
            in_position = True
        elif in_position and exit_signal.iloc[i]:
            in_position = False
        position.iloc[i] = int(in_position)
    return position


def rsi_mean_reversion(df: pd.DataFrame, period: int = 14, oversold: int = 30, exit_level: int = 50) -> pd.Series:
    """Contrarian: buy when RSI drops below `oversold` (price fell hard, fast), hold until RSI
    recovers above `exit_level`. Opposite hypothesis to the trend-followers above — bets that
    sharp short-term drops tend to bounce rather than persist.
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    entry_signal = rsi < oversold
    exit_signal = rsi > exit_level

    position = pd.Series(0, index=df.index)
    in_position = False
    for i in range(len(df)):
        if not in_position and entry_signal.iloc[i]:
            in_position = True
        elif in_position and exit_signal.iloc[i]:
            in_position = False
        position.iloc[i] = int(in_position)
    return position
