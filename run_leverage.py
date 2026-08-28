"""Leverage sweep: same SMA 50/200 strategy, various leverage levels, real historical funding cost."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.data import fetch_daily_funding_rate, fetch_ohlcv
from src.leverage_backtest import run_leveraged_backtest
from src.strategies import sma_crossover

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2019-09-01")
    funding = fetch_daily_funding_rate("BTC/USDT", since="2019-09-01")
    print(f"Loaded {len(df)} daily candles and {len(funding)} days of real funding-rate history")
    print(f"Average funding: {funding.mean()*100:.4f}%/day  (annualized ~{funding.mean()*365*100:.2f}%, paid by longs when positive)\n")

    position = sma_crossover(df, fast=50, slow=200)

    leverages = [1, 2, 3, 5, 10, 20]
    results = {}
    print(f"{'leverage':>9} {'total_return%':>14} {'cagr%':>8} {'sharpe':>8} {'max_dd%':>9} {'liquidations':>13}")
    for lev in leverages:
        result = run_leveraged_backtest(df, position, leverage=lev, funding_daily=funding)
        results[f"{lev}x"] = result
        m = result.metrics
        print(f"{lev:>8}x {m['total_return_pct']:>14} {m['cagr_pct']:>8} {m['sharpe']:>8} {m['max_drawdown_pct']:>9} {m['n_liquidations']:>13}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "leverage_metrics.json", "w") as f:
        json.dump({k: v.metrics for k, v in results.items()}, f, indent=2)

    plt.figure(figsize=(11, 6))
    for name, result in results.items():
        plt.plot(result.equity_curve.index, result.equity_curve.values, label=name)
        for d in result.liquidation_dates:
            plt.axvline(d, color="red", alpha=0.08, linewidth=1)
    plt.yscale("log")
    plt.title("SMA 50/200 on BTC/USDT — leverage sweep (log scale, red lines = liquidation events)")
    plt.xlabel("date")
    plt.ylabel("equity (log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "leverage_equity_curve.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'leverage_equity_curve.png'} and {RESULTS_DIR / 'leverage_metrics.json'}")


if __name__ == "__main__":
    main()
