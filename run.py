"""Entry point: fetch BTC/USDT daily data, backtest a couple of strategies, report results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import buy_and_hold, sma_crossover

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")
    print(f"Loaded {len(df)} daily candles: {df.index[0].date()} → {df.index[-1].date()}")

    runs = {
        "buy_and_hold": run_backtest(df, buy_and_hold(df)),
        "sma_20_50": run_backtest(df, sma_crossover(df, fast=20, slow=50)),
        "sma_50_200": run_backtest(df, sma_crossover(df, fast=50, slow=200)),
    }

    print(f"\n{'strategy':<14} {'total_return%':>14} {'cagr%':>8} {'sharpe':>8} {'max_dd%':>9} {'trades':>7} {'win%':>6}")
    for name, result in runs.items():
        m = result.metrics
        print(
            f"{name:<14} {m['total_return_pct']:>14} {m['cagr_pct']:>8} {m['sharpe']:>8} "
            f"{m['max_drawdown_pct']:>9} {m['n_trades']:>7} {m['win_rate_pct']:>6}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump({name: r.metrics for name, r in runs.items()}, f, indent=2)

    plt.figure(figsize=(11, 6))
    for name, result in runs.items():
        plt.plot(result.equity_curve.index, result.equity_curve.values, label=name)
    plt.yscale("log")
    plt.title("BTC/USDT — equity curve by strategy (log scale, starting equity = 1)")
    plt.xlabel("date")
    plt.ylabel("equity (log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "equity_curve.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'equity_curve.png'} and {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
