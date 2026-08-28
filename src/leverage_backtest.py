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

Also includes `compute_target_leverage` / `run_vol_targeted_backtest`: a volatility-targeting
extension that replaces a single fixed leverage multiplier with one recomputed daily from
recent realized volatility. See `compute_target_leverage`'s docstring for its own assumptions.
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


def compute_target_leverage(
    df: pd.DataFrame,
    target_annual_vol: float,
    vol_lookback: int = 20,
    min_leverage: float = 0.0,
    max_leverage: float = 5.0,
) -> pd.Series:
    """Volatility-targeting leverage schedule: size leverage inversely to recent realized
    volatility so the position aims for roughly constant *risk* per day, instead of a flat
    multiplier applied blindly regardless of market conditions. Standard technique ("vol
    targeting"), used e.g. by risk-parity and managed-futures funds and by leveraged-vol-target
    ETFs.

    Assumptions, stated plainly (same spirit as this module's other simplifications):
    - Realized vol is a rolling `vol_lookback`-day standard deviation of daily close-to-close
      pct returns, annualized by sqrt(365) (same annualization used everywhere else in this repo).
      This is a simple, standard estimator -- not an EWMA or GARCH model, which would react
      faster to a vol regime change at the cost of more noise.
    - Leverage for day t is computed from data up to and including day t-1 (shifted one day,
      the same lag convention `run_leveraged_backtest` and `run_backtest` already use for the
      position signal) -- you can only size against volatility you've already observed, not
      today's.
    - `leverage(t) = target_annual_vol / realized_vol(t)`, clipped to `[min_leverage,
      max_leverage]`. Unclipped, a very calm rolling window would imply arbitrarily large
      leverage, which nobody should actually run; the cap is a deliberate ceiling, not a
      byproduct of the vol estimate.
    - This targets *volatility*, not drawdown or tail risk directly. Realized vol is backward-
      looking and lags a regime change -- it won't size down until a few violent days have
      already pushed the rolling window up, so it dampens ordinary chop more than it protects
      against a single overnight gap or crash. It is not a substitute for a hard leverage cap
      or for staying under a strategy's liquidation-free leverage range.
    """
    daily_return = df["close"].pct_change()
    realized_vol = daily_return.rolling(vol_lookback).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    raw_leverage = target_annual_vol / realized_vol.replace(0, np.nan)
    return raw_leverage.shift(1).clip(lower=min_leverage, upper=max_leverage)


def run_vol_targeted_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    target_leverage: pd.Series,
    funding_daily: pd.Series | None = None,
    fee_rate: float = 0.0004,
    maintenance_margin_rate: float = 0.005,
) -> LeveragedBacktestResult:
    """Same mechanics as `run_leveraged_backtest` (daily-rebalanced exposure, intrabar
    liquidation checked against the day's low, real historical funding cost) but with a
    per-day leverage multiplier instead of one fixed value for the whole backtest -- see
    `compute_target_leverage` for how that schedule is built and what it assumes. A day where
    the schedule lands at 0 leverage is treated as fully sized out (flat exposure, no funding,
    not liquidatable) rather than skipped.
    """
    lagged_position = position.shift(1).fillna(0)
    leverage_series = target_leverage.reindex(df.index).fillna(0.0).clip(lower=0.0)
    funding = (
        funding_daily.reindex(df.index).fillna(0)
        if funding_daily is not None
        else pd.Series(0.0, index=df.index)
    )

    equity = 1.0
    in_position = False
    equity_values = []
    liquidation_dates = []
    leverage_used = []
    n_trades = 0
    prev_close = None

    for i, date in enumerate(df.index):
        o, l, c = df["open"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
        want_long = bool(lagged_position.iloc[i])
        lev = float(leverage_series.iloc[i])
        ref_price = prev_close if (in_position and prev_close is not None) else o

        if want_long and not in_position:
            in_position = True
            n_trades += 1
            equity *= 1 - fee_rate * lev
            ref_price = o

        if in_position:
            if lev > 0:
                liq_adverse_move = max(1.0 / lev - maintenance_margin_rate, 0.0)
                liq_residual_fraction = min(max(lev * maintenance_margin_rate, 0.0), 1.0)
                adverse_move = (ref_price - l) / ref_price
                if adverse_move >= liq_adverse_move:
                    equity *= liq_residual_fraction
                    liquidation_dates.append(date)
                    in_position = False
                else:
                    day_return = (c - ref_price) / ref_price
                    equity *= 1 + lev * day_return - lev * funding.iloc[i]
                    leverage_used.append(lev)
                    if not want_long:
                        equity *= 1 - fee_rate * lev
                        in_position = False
            else:
                # vol target scaled exposure to 0 for the day -- flat, no P&L, no liquidation risk.
                leverage_used.append(0.0)
                if not want_long:
                    in_position = False

        equity_values.append(equity)
        prev_close = c

    equity_curve = pd.Series(equity_values, index=df.index)
    metrics = _compute_metrics(equity_curve, n_trades, liquidation_dates)
    metrics["avg_leverage_when_in_position"] = (
        round(float(np.mean(leverage_used)), 2) if leverage_used else 0.0
    )
    return LeveragedBacktestResult(equity_curve=equity_curve, metrics=metrics, liquidation_dates=liquidation_dates)
