"""Vectorized backtest engine: position signal + price series in, equity curve + metrics out."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 365  # crypto trades every day, unlike equities


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    metrics: dict


def run_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    fee_rate: float = 0.0005,  # 5 bps per side — a reasonable taker-fee assumption
    execution_lag: int = 1,
) -> BacktestResult:
    """execution_lag=1 (default): positions are applied with a 1-bar lag, so a signal confirmed
    at today's close only starts earning tomorrow's return. This is the realistic, tradeable
    convention -- you can't act on a close before it happens, so the very day a breakout closes
    high enough to confirm the signal, that day's own move is already behind you.

    execution_lag=0: the position applies to the SAME day its signal is computed, so it captures
    that day's own return too -- including the move that caused the breakout to confirm in the
    first place. This isn't something you could actually place a same-day trade to capture (you
    don't know the close confirms the pattern until the session is over), so treat it as a
    best-case / upper-bound comparison against the lag=1 number, not a live-tradeable rule.

    Every position *change* pays `fee_rate` once, approximating round-trip exchange fees.
    """
    price = df["close"]
    daily_return = price.pct_change().fillna(0)

    lagged_position = position.shift(execution_lag).fillna(0)
    position_change = lagged_position.diff().abs().fillna(lagged_position.abs())
    fees = position_change * fee_rate

    strategy_return = lagged_position * daily_return - fees
    equity_curve = (1 + strategy_return).cumprod()

    metrics = _compute_metrics(equity_curve, strategy_return, position_change)
    return BacktestResult(equity_curve=equity_curve, metrics=metrics)


def _compute_metrics(equity_curve: pd.Series, returns: pd.Series, position_change: pd.Series) -> dict:
    n_days = len(equity_curve)
    total_return = equity_curve.iloc[-1] - 1
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = (equity_curve.iloc[-1] ** (1 / years) - 1) if years > 0 and equity_curve.iloc[-1] > 0 else float("nan")

    vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (returns.mean() * TRADING_DAYS_PER_YEAR) / vol if vol > 0 else float("nan")

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    n_trades = int((position_change > 0).sum())
    winning_days = int((returns > 0).sum())
    active_days = int((returns != 0).sum())
    win_rate = winning_days / active_days if active_days > 0 else float("nan")

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 2),
        "n_days": n_days,
    }
