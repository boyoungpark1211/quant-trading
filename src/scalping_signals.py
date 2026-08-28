"""Entry-signal generators for the scalping backtest. Each returns a boolean Series aligned to
df.index: True on bars where a long entry should be attempted (the backtest engine ignores the
signal while already in a position, so no state-machine logic belongs here).
"""

import pandas as pd


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def rsi_oversold_signal(df: pd.DataFrame, period: int = 2, threshold: float = 15.0) -> pd.Series:
    """Mean-reversion hypothesis: enter on a sharp short-term selloff.

    Rejected on BTC/KRW 1m data: 0/48 parameter combinations were profitable on the train
    split (see results/scalp_sweep_train.csv), and the best train configs stayed negative
    on the held-out test split too. Kept here for reference/comparison, not as a live candidate.
    """
    r = rsi(df["close"], period)
    return r < threshold


def momentum_breakout_signal(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Momentum hypothesis: enter when price makes a new `lookback`-bar high -- ride a short
    burst rather than bet on a reversal. Mirrors the daily Donchian breakout logic, which
    outperformed daily mean-reversion in the swing-strategy backtests (see compare_all.py).
    """
    rolling_high = df["close"].rolling(lookback).max().shift(1)
    return df["close"] >= rolling_high


def momentum_spike_signal(
    df: pd.DataFrame, roc_lookback_bars: int = 2, min_move_pct: float = 0.002, volume_mult: float | None = 2.0
) -> pd.Series:
    """A genuinely different hypothesis from momentum_breakout_signal above: that one fires on
    ANY new N-bar high, including a slow grind up over N minutes. This one fires only on a sharp
    price move within a SHORT number of bars (rate-of-change spike) -- an actual sudden surge,
    not just "currently the highest price recently seen." Optionally requires the move to be
    volume-confirmed (volume above its own rolling average by volume_mult), since a real surge
    should trade on above-average size, not thin/illiquid drift.
    """
    roc = df["close"].pct_change(roc_lookback_bars)
    signal = roc >= min_move_pct
    if volume_mult is not None:
        avg_volume = df["volume"].rolling(20).mean()
        signal = signal & (df["volume"] > avg_volume * volume_mult)
    return signal.fillna(False)


def with_trend_filter(
    signal: pd.Series, df: pd.DataFrame, ema_period: int, require_uptrend: bool = True
) -> pd.Series:
    """AND a signal with a longer-EMA trend filter -- e.g. only take momentum entries that agree
    with the broader short-term trend, to avoid buying a breakout inside a bigger downtrend.
    """
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    trend_ok = (df["close"] > ema) if require_uptrend else (df["close"] < ema)
    return signal & trend_ok
