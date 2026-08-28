"""₩10,000 virtual scalping run: compare RSI(2) mean-reversion vs. momentum breakout on Upbit
BTC/KRW 1-minute bars, same bracket-order execution for both.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.data import fetch_ohlcv
from src.scalping_backtest import run_scalp_backtest
from src.scalping_signals import momentum_breakout_signal, rsi_oversold_signal

RESULTS_DIR = Path(__file__).resolve().parent / "results"
STARTING_CAPITAL_KRW = 10_000


def main() -> None:
    df = fetch_ohlcv("upbit", "BTC/KRW", timeframe="1m", since="2026-07-10")
    print(f"Loaded {len(df)} 1-minute candles: {df.index[0]} -> {df.index[-1]}\n")

    runs = {
        "rsi_mean_reversion": run_scalp_backtest(
            df, rsi_oversold_signal(df, period=2, threshold=15.0), starting_capital=STARTING_CAPITAL_KRW
        ),
        "momentum_breakout": run_scalp_backtest(
            df, momentum_breakout_signal(df, lookback=10), starting_capital=STARTING_CAPITAL_KRW
        ),
    }

    print(f"{'strategy':<20}{'return%':>10}{'final':>10}{'trades':>8}{'win%':>7}{'tp':>5}{'sl':>5}{'timeout':>9}")
    for name, result in runs.items():
        m = result.metrics
        print(
            f"{name:<20}{m['total_return_pct']:>10}{m['final_capital']:>10.0f}{m['n_trades']:>8}"
            f"{m['win_rate_pct']:>7}{m['tp_hits']:>5}{m['sl_hits']:>5}{m['timeouts']:>9}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "scalp_metrics.json", "w") as f:
        json.dump({k: v.metrics for k, v in runs.items()}, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(11, 6))
    for name, result in runs.items():
        plt.plot(result.equity_curve.index, result.equity_curve.values, label=name)
    plt.axhline(STARTING_CAPITAL_KRW, color="black", linewidth=0.8, linestyle="--", label="starting capital")
    plt.title(f"Scalping: RSI mean-reversion vs. momentum breakout (start {STARTING_CAPITAL_KRW:,} KRW)")
    plt.ylabel("KRW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scalp_result.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'scalp_result.png'}")


if __name__ == "__main__":
    main()
