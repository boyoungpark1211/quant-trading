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


def _positions_from_signals(entry_signal: pd.Series, exit_signal: pd.Series) -> pd.Series:
    """Shared state machine: flip long on `entry_signal`, flip flat on `exit_signal`, hold
    otherwise. Factored out of the filtered Donchian variants below so each one only has to
    define how it computes entry_signal/exit_signal, not re-implement the loop.
    """
    position = pd.Series(0, index=entry_signal.index)
    in_position = False
    for i in range(len(entry_signal)):
        if not in_position and entry_signal.iloc[i]:
            in_position = True
        elif in_position and exit_signal.iloc[i]:
            in_position = False
        position.iloc[i] = int(in_position)
    return position


def _atr_pct(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    """Average True Range as a % of price -- scale-free, so a threshold on it means the same
    thing whether BTC is at $4k (2018) or $100k+ (2025), unlike a threshold on raw ATR or
    raw rolling return std.
    """
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(atr_period).mean()
    return atr / df["close"]


def donchian_breakout_vol_filtered(
    df: pd.DataFrame,
    entry_lookback: int = 25,
    exit_lookback: int = 10,
    atr_period: int = 14,
    vol_rank_lookback: int = 252,
    vol_rank_threshold: float = 0.5,
) -> pd.Series:
    """Donchian breakout (same rule as `donchian_breakout`) gated by a volatility-regime filter:
    only take a breakout entry when recent realized volatility (ATR%) ranks at or above
    `vol_rank_threshold` (a percentile, 0-1) within its own trailing `vol_rank_lookback`-day
    history.

    Hypothesis: a slow grinding chop (like the 2022 Luna/FTX bear) tends to have LOWER realized
    volatility than a genuine trending or crashing move (like the 2020 COVID V-crash, or a real
    breakout into a new bull leg) -- both because chop lacks the sustained directional
    participation of a real move, and because the whipsaw losses come from many small false
    breakouts, which happen more often when the market isn't moving enough to sustain a real
    trend. Gating entries on "vol is elevated relative to its own recent history" should
    suppress low-conviction breakouts without touching the genuine ones.

    A rolling *rank* is used instead of a fixed ATR% cutoff because BTC's absolute volatility
    has changed by orders of magnitude across 2018-2026 -- a fixed number picked on one era would
    be meaningless (too strict or too loose) in another.
    """
    entry_signal_raw = df["close"] >= df["close"].rolling(entry_lookback).max().shift(1)
    exit_signal = df["close"] <= df["close"].rolling(exit_lookback).min().shift(1)

    vol = _atr_pct(df, atr_period)
    vol_rank = vol.rolling(vol_rank_lookback).apply(lambda w: (w <= w[-1]).mean(), raw=True)
    entry_signal = entry_signal_raw & (vol_rank >= vol_rank_threshold)

    return _positions_from_signals(entry_signal.fillna(False), exit_signal)


def donchian_breakout_margin_filtered(
    df: pd.DataFrame,
    entry_lookback: int = 25,
    exit_lookback: int = 10,
    min_breakout_margin_pct: float = 1.0,
) -> pd.Series:
    """Donchian breakout gated by a breakout-strength filter: only enter when the close clears
    the prior `entry_lookback`-day high by at least `min_breakout_margin_pct` percent, not just
    barely poking above it.

    Hypothesis: in chop, price repeatedly grazes the top of its recent range by a hair and falls
    back -- each graze is a "new high" by the letter of the Donchian rule but not a decisive move.
    Requiring a real margin should filter out those marginal breakouts while leaving genuine
    trend/crash moves (which tend to clear the prior range convincingly, not by a few bps) intact.
    """
    prior_high = df["close"].rolling(entry_lookback).max().shift(1)
    entry_signal = df["close"] >= prior_high * (1 + min_breakout_margin_pct / 100)
    exit_signal = df["close"] <= df["close"].rolling(exit_lookback).min().shift(1)

    return _positions_from_signals(entry_signal.fillna(False), exit_signal)


def donchian_breakout_confirmed(
    df: pd.DataFrame,
    entry_lookback: int = 25,
    exit_lookback: int = 10,
    confirm_bars: int = 2,
) -> pd.Series:
    """Donchian breakout gated by a confirmation-bar filter: only enter once the close has stayed
    at or above the prior `entry_lookback`-day high for `confirm_bars` consecutive days, not on
    the first poke above it.

    Hypothesis: a single-day breakout in a choppy market often reverses immediately (the false
    starts driving 2022's whipsaws); a move that holds for a few extra days is more likely to be
    a real trend. Trades a slightly worse (later, less favorable) entry price for skipping
    single-bar fake-outs.
    """
    prior_high = df["close"].rolling(entry_lookback).max().shift(1)
    above_high = df["close"] >= prior_high
    entry_signal = above_high.rolling(confirm_bars).sum() >= confirm_bars
    exit_signal = df["close"] <= df["close"].rolling(exit_lookback).min().shift(1)

    return _positions_from_signals(entry_signal.fillna(False), exit_signal)


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

    Rejected after walk-forward validation: looked promising in-sample (94% of a 36-combo train
    sweep beat buy&hold, best Sharpe 1.10) but all four validated candidates collapsed on the
    held-out test split (Sharpe 0.35-0.60, well below both buy&hold and donchian_25_10). Kept
    here for reference/comparison, not as a live candidate -- see validate_bollinger_squeeze.py.

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

    return _positions_from_signals(entry_signal.fillna(False), exit_signal)


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
