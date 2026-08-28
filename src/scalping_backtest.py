"""Bracket-order scalping backtest: given an entry signal, exit at a fixed take-profit,
stop-loss, or after a max holding time. This is closer to how a scalper actually trades
(bracket orders on a fill) than the signal-flip logic used for the daily swing strategies.

Entry signal generation lives in `scalping_signals.py` and is passed in as a boolean array,
so different hypotheses (mean-reversion, momentum, ...) share the exact same execution
mechanics — the only fair way to compare them.

Numpy-array loop (not pandas .iloc) — this gets called dozens/hundreds of times in a parameter
sweep, and .iloc row access is slow enough to matter at that scale.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScalpResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict


def run_scalp_backtest(
    df: pd.DataFrame,
    entry_signal: pd.Series,
    starting_capital: float,
    take_profit_pct: float = 0.003,
    stop_loss_pct: float = 0.002,
    max_hold_bars: int = 30,
    fee_rate: float = 0.0005,  # Upbit KRW-market taker fee, per side
) -> ScalpResult:
    """entry_signal: boolean Series aligned to df.index -- True on bars where a long should be
    opened (if not already in one). Exit at +take_profit_pct, -stop_loss_pct, or after
    max_hold_bars bars, whichever comes first. Fee is charged on both entry and exit.
    """
    signal_arr = entry_signal.reindex(df.index).fillna(False).to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index

    n = len(df)
    capital = starting_capital
    equity_values = np.empty(n)
    trades = []

    in_position = False
    entry_price = entry_idx = target = stop = 0.0

    for i in range(n):
        if not in_position:
            if signal_arr[i]:
                in_position = True
                entry_price = c[i]
                entry_idx = i
                target = entry_price * (1 + take_profit_pct)
                stop = entry_price * (1 - stop_loss_pct)
                capital *= 1 - fee_rate
        else:
            bars_held = i - entry_idx
            exit_price = None
            reason = None
            if h[i] >= target:
                exit_price = target
                reason = "take_profit"
            elif l[i] <= stop:
                exit_price = stop
                reason = "stop_loss"
            elif bars_held >= max_hold_bars:
                exit_price = c[i]
                reason = "timeout"

            if exit_price is not None:
                trade_return = (exit_price - entry_price) / entry_price
                capital *= 1 + trade_return
                capital *= 1 - fee_rate
                trades.append(
                    {
                        "entry_time": idx[entry_idx],
                        "exit_time": idx[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "bars_held": bars_held,
                        "return_pct": trade_return * 100,
                        "reason": reason,
                    }
                )
                in_position = False

        equity_values[i] = capital

    equity_curve = pd.Series(equity_values, index=df.index)
    trades_df = pd.DataFrame(trades)
    metrics = _compute_metrics(equity_curve, trades_df, starting_capital)
    return ScalpResult(trades=trades_df, equity_curve=equity_curve, metrics=metrics)


def _compute_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame, starting_capital: float) -> dict:
    final = equity_curve.iloc[-1]
    total_return_pct = (final / starting_capital - 1) * 100
    n_trades = len(trades_df)

    if n_trades == 0:
        return {
            "total_return_pct": round(total_return_pct, 2),
            "n_trades": 0,
            "win_rate_pct": float("nan"),
            "avg_return_per_trade_pct": float("nan"),
            "final_capital": round(final, 0),
            "tp_hits": 0,
            "sl_hits": 0,
            "timeouts": 0,
        }

    wins = (trades_df["return_pct"] > 0).sum()
    return {
        "total_return_pct": round(total_return_pct, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(wins / n_trades * 100, 1),
        "avg_return_per_trade_pct": round(trades_df["return_pct"].mean(), 4),
        "final_capital": round(final, 0),
        "tp_hits": int((trades_df["reason"] == "take_profit").sum()),
        "sl_hits": int((trades_df["reason"] == "stop_loss").sum()),
        "timeouts": int((trades_df["reason"] == "timeout").sum()),
    }
