"""Bracket-order scalping backtest: enter on a short-term RSI extreme, exit at a fixed
take-profit, stop-loss, or after a max holding time. This is closer to how a scalper actually
trades (bracket orders on a fill) than the signal-flip logic used for the daily swing strategies.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScalpResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def run_scalp_backtest(
    df: pd.DataFrame,
    starting_capital: float,
    rsi_period: int = 2,
    rsi_entry: float = 15.0,
    take_profit_pct: float = 0.003,
    stop_loss_pct: float = 0.002,
    max_hold_bars: int = 30,
    fee_rate: float = 0.0005,  # Upbit KRW-market taker fee, per side — from exchange.describe()['options']['tradingFeesByQuoteCurrency']['KRW']
) -> ScalpResult:
    """Long-only mean-reversion scalp: buy when RSI(rsi_period) drops below rsi_entry (a sharp,
    short-term selloff), exit at +take_profit_pct, -stop_loss_pct, or after max_hold_bars bars,
    whichever comes first. Fee is charged on both entry and exit.
    """
    signal_rsi = rsi(df["close"], rsi_period)

    capital = starting_capital
    equity_values = []
    trades = []

    in_position = False
    entry_price = entry_idx = target = stop = None

    for i in range(len(df)):
        o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]

        if not in_position:
            r = signal_rsi.iloc[i]
            if not np.isnan(r) and r < rsi_entry:
                in_position = True
                entry_price = c
                entry_idx = i
                target = entry_price * (1 + take_profit_pct)
                stop = entry_price * (1 - stop_loss_pct)
                capital *= 1 - fee_rate
        else:
            bars_held = i - entry_idx
            exit_price = None
            reason = None
            if h >= target:
                exit_price = target
                reason = "take_profit"
            elif l <= stop:
                exit_price = stop
                reason = "stop_loss"
            elif bars_held >= max_hold_bars:
                exit_price = c
                reason = "timeout"

            if exit_price is not None:
                trade_return = (exit_price - entry_price) / entry_price
                capital *= 1 + trade_return
                capital *= 1 - fee_rate
                trades.append(
                    {
                        "entry_time": df.index[entry_idx],
                        "exit_time": df.index[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "bars_held": bars_held,
                        "return_pct": trade_return * 100,
                        "reason": reason,
                    }
                )
                in_position = False

        equity_values.append(capital)

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
