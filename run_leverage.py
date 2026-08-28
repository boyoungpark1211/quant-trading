"""Leverage sweep: pick a strategy, various leverage levels, real historical funding cost."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.data import fetch_daily_funding_rate, fetch_ohlcv
from src.leverage_backtest import run_leveraged_backtest
from src.strategies import donchian_breakout, rsi_mean_reversion, sma_crossover

RESULTS_DIR = Path(__file__).resolve().parent / "results"

STRATEGIES = {
    "sma_50_200": lambda df: sma_crossover(df, fast=50, slow=200),
    "sma_20_50": lambda df: sma_crossover(df, fast=20, slow=50),
    "donchian_20_10": lambda df: donchian_breakout(df, entry_lookback=20, exit_lookback=10),
    # Walk-forward-validated upgrades over 20/10 (see sweep_donchian.py / validate_donchian_upgrade.py):
    # 25/10 is the best all-around (test Sharpe 1.46, CAGR 46%), 70/20 trades some Sharpe for
    # much lower drawdown (test Sharpe 0.97, CAGR 26.6%, fewer/more selective trades).
    "donchian_25_10": lambda df: donchian_breakout(df, entry_lookback=25, exit_lookback=10),
    "donchian_70_20": lambda df: donchian_breakout(df, entry_lookback=70, exit_lookback=20),
    "rsi_mean_reversion": lambda df: rsi_mean_reversion(df, period=14, oversold=30, exit_level=50),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES.keys(), default="sma_50_200")
    args = parser.parse_args()

    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2019-09-01")
    funding = fetch_daily_funding_rate("BTC/USDT", since="2019-09-01")
    print(f"Strategy: {args.strategy}")
    print(f"Loaded {len(df)} daily candles and {len(funding)} days of real funding-rate history")
    print(f"Average funding: {funding.mean()*100:.4f}%/day  (annualized ~{funding.mean()*365*100:.2f}%, paid by longs when positive)\n")

    position = STRATEGIES[args.strategy](df)

    leverages = [1, 2, 3, 5, 10, 20]
    results = {}
    print(f"{'leverage':>9} {'total_return%':>14} {'cagr%':>8} {'sharpe':>8} {'max_dd%':>9} {'liquidations':>13}")
    for lev in leverages:
        result = run_leveraged_backtest(df, position, leverage=lev, funding_daily=funding)
        results[f"{lev}x"] = result
        m = result.metrics
        print(f"{lev:>8}x {m['total_return_pct']:>14} {m['cagr_pct']:>8} {m['sharpe']:>8} {m['max_drawdown_pct']:>9} {m['n_liquidations']:>13}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"leverage_metrics_{args.strategy}.json", "w") as f:
        json.dump({k: v.metrics for k, v in results.items()}, f, indent=2)

    plt.figure(figsize=(11, 6))
    for name, result in results.items():
        plt.plot(result.equity_curve.index, result.equity_curve.values, label=name)
        for d in result.liquidation_dates:
            plt.axvline(d, color="red", alpha=0.08, linewidth=1)
    plt.yscale("log")
    plt.title(f"{args.strategy} on BTC/USDT — leverage sweep (log scale, red lines = liquidation events)")
    plt.xlabel("date")
    plt.ylabel("equity (log)")
    plt.legend()
    plt.tight_layout()
    out_png = RESULTS_DIR / f"leverage_equity_curve_{args.strategy}.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png} and results/leverage_metrics_{args.strategy}.json")


if __name__ == "__main__":
    main()
