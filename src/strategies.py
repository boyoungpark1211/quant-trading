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


def bollinger_squeeze_breakout(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    squeeze_lookback: int = 120,
    squeeze_percentile: float = 0.20,
) -> pd.Series:
    """Volatility-regime breakout: go long when price breaks above the upper Bollinger Band
    *and* that band was unusually narrow (a "squeeze") just before the break; exit when price
    falls back below the middle band (the SMA).

    Different mechanism from donchian_breakout: Donchian reacts to a raw N-day price extreme
    regardless of how volatile the recent past was. This strategy instead conditions the
    breakout on a volatility *regime* -- it only takes breakouts that follow an unusually quiet
    period (bandwidth in the bottom `squeeze_percentile` of its own trailing
    `squeeze_lookback`-day distribution), on the hypothesis that a breakout out of compressed
    volatility carries more signal (a real regime change) than a breakout during already-wide,
    choppy bands (more likely noise). `bb_period`/`bb_std` are left at their textbook-standard
    values (20-day SMA, 2 std devs) rather than swept, since they define what "the bands" even
    are; `squeeze_lookback` and `squeeze_percentile` are this hypothesis's actual free
    parameters and are the ones the sweep grid-searches.

    All bands/percentile-rank inputs are shifted by one day before comparing against today's
    close, so the entry/exit decision at time t only ever uses information known through t-1
    (today's close isn't part of today's own trigger threshold) -- same convention as
    donchian_breakout's `.shift(1)` on its rolling max/min.
    """
    close = df["close"]
    mid = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    upper = mid + bb_std * std
    lower = mid - bb_std * std
    bandwidth = (upper - lower) / mid

    # Percentile rank of each day's bandwidth within its own trailing window: fraction of the
    # last `squeeze_lookback` days with bandwidth <= today's. Low value = unusually narrow bands.
    bandwidth_percentile = bandwidth.rolling(squeeze_lookback).apply(
        lambda w: (w <= w[-1]).mean(), raw=True
    )

    was_squeezed = (bandwidth_percentile <= squeeze_percentile).shift(1, fill_value=False)
    breakout_up = close > upper.shift(1)
    entry_signal = was_squeezed & breakout_up
    exit_signal = close <= mid.shift(1)

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
