"""Leverage-aware backtest: daily-rebalanced leveraged exposure, intrabar liquidation checks,
and real historical funding cost.

Simplifications, stated plainly:
- Leverage is applied fresh each day against the previous close (like a daily-rebalanced
  leveraged product), not tracked against a single fixed entry price for the whole holding
  period. This is the standard approximation used for leveraged ETF/perpetual backtests and
  avoids path-dependent entry-price drift bugs.
- Maintenance margin is a single flat rate (default 0.5%), not Binance's real tiered
  bracket table (which varies by notional size and leverage). This is fine for illustrating
  liquidation risk directionally; it is not a precise liquidation-price calculator. Check
  Binance's published maintenance margin table before sizing a real position.
- Liquidation is checked against the day's *low* (for a long), which is closer to reality
  than a close-only check but still coarser than tick-level mark price.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 365


@dataclass
class LeveragedBacktestResult:
    equity_curve: pd.Series
    metrics: dict
    liquidation_dates: list


def run_leveraged_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    leverage: float,
    funding_daily: pd.Series | None = None,
    fee_rate: float = 0.0004,
    maintenance_margin_rate: float = 0.005,
) -> LeveragedBacktestResult:
    lagged_position = position.shift(1).fillna(0)
    funding = (
        funding_daily.reindex(df.index).fillna(0)
        if funding_daily is not None
        else pd.Series(0.0, index=df.index)
    )

    liq_adverse_move = max(1.0 / leverage - maintenance_margin_rate, 0.0)
    liq_residual_fraction = min(max(leverage * maintenance_margin_rate, 0.0), 1.0)

    equity = 1.0
    in_position = False
    equity_values = []
    liquidation_dates = []
    n_trades = 0
    prev_close = None

    for i, date in enumerate(df.index):
        o, l, c = df["open"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
        want_long = bool(lagged_position.iloc[i])
        ref_price = prev_close if (in_position and prev_close is not None) else o

        if want_long and not in_position:
            in_position = True
            n_trades += 1
            equity *= 1 - fee_rate * leverage
            ref_price = o

        if in_position:
            adverse_move = (ref_price - l) / ref_price
            if adverse_move >= liq_adverse_move:
                equity *= liq_residual_fraction
                liquidation_dates.append(date)
                in_position = False
            else:
                day_return = (c - ref_price) / ref_price
                equity *= 1 + leverage * day_return - leverage * funding.iloc[i]
                if not want_long:
                    equity *= 1 - fee_rate * leverage
                    in_position = False

        equity_values.append(equity)
        prev_close = c

    equity_curve = pd.Series(equity_values, index=df.index)
    metrics = _compute_metrics(equity_curve, n_trades, liquidation_dates)
    return LeveragedBacktestResult(equity_curve=equity_curve, metrics=metrics, liquidation_dates=liquidation_dates)


def _compute_metrics(equity_curve: pd.Series, n_trades: int, liquidation_dates: list) -> dict:
    n_days = len(equity_curve)
    years = n_days / TRADING_DAYS_PER_YEAR
    final = equity_curve.iloc[-1]
    total_return = final - 1
    cagr = (final ** (1 / years) - 1) if years > 0 and final > 0 else -1.0

    daily_ret = equity_curve.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (daily_ret.mean() * TRADING_DAYS_PER_YEAR) / vol if vol > 0 else float("nan")

    running_max = equity_curve.cummax()
    max_drawdown = (equity_curve / running_max - 1).min()

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "n_trades": n_trades,
        "n_liquidations": len(liquidation_dates),
        "final_equity": round(final, 4),
    }
