"""Compare all strategies over the full period AND a walk-forward split (train vs. unseen test)
so a strategy that only worked in-sample gets caught before anyone trusts it.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import buy_and_hold, donchian_breakout, rsi_mean_reversion, sma_crossover

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLIT_DATE = "2023-01-01"  # train: 2018-01-01..2022-12-31, test: 2023-01-01..today


def strategies(df: pd.DataFrame) -> dict:
    return {
        "buy_and_hold": buy_and_hold(df),
        "sma_20_50": sma_crossover(df, 20, 50),
        "sma_50_200": sma_crossover(df, 50, 200),
        "donchian_20_10": donchian_breakout(df, entry_lookback=20, exit_lookback=10),
        "rsi_mean_reversion": rsi_mean_reversion(df, period=14, oversold=30, exit_level=50),
    }


def print_table(title: str, metrics_by_strategy: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"{'strategy':<20} {'total_return%':>14} {'cagr%':>8} {'sharpe':>8} {'max_dd%':>9} {'trades':>7} {'win%':>6}")
    for name, m in metrics_by_strategy.items():
        print(
            f"{name:<20} {m['total_return_pct']:>14} {m['cagr_pct']:>8} {m['sharpe']:>8} "
            f"{m['max_drawdown_pct']:>9} {m['n_trades']:>7} {m['win_rate_pct']:>6}"
        )


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")
    positions = strategies(df)

    # full-period
    full_results = {name: run_backtest(df, pos) for name, pos in positions.items()}
    print_table("Full period 2018-2026", {n: r.metrics for n, r in full_results.items()})

    # walk-forward: compute the signal on the FULL series (so indicators like SMA/RSI have proper
    # warm-up history), then backtest only over each sub-period — this is what a strategy running
    # live in 2023 would have actually seen, not a fresh series that starts blind in 2023.
    train_df = df[df.index < SPLIT_DATE]
    test_df = df[df.index >= SPLIT_DATE]

    train_results, test_results = {}, {}
    for name, pos in positions.items():
        train_results[name] = run_backtest(train_df, pos.reindex(train_df.index))
        test_results[name] = run_backtest(test_df, pos.reindex(test_df.index))

    print_table(f"Train (2018 .. {SPLIT_DATE})", {n: r.metrics for n, r in train_results.items()})
    print_table(f"Test / out-of-sample ({SPLIT_DATE} .. 2026)", {n: r.metrics for n, r in test_results.items()})

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "compare_all_metrics.json", "w") as f:
        json.dump(
            {
                "full": {n: r.metrics for n, r in full_results.items()},
                "train": {n: r.metrics for n, r in train_results.items()},
                "test": {n: r.metrics for n, r in test_results.items()},
            },
            f,
            indent=2,
        )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for name, result in train_results.items():
        axes[0].plot(result.equity_curve.index, result.equity_curve.values, label=name)
    axes[0].set_yscale("log")
    axes[0].set_title(f"Train: 2018 .. {SPLIT_DATE}")
    axes[0].legend(fontsize=8)

    for name, result in test_results.items():
        axes[1].plot(result.equity_curve.index, result.equity_curve.values, label=name)
    axes[1].set_yscale("log")
    axes[1].set_title(f"Test (out-of-sample): {SPLIT_DATE} .. 2026")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "walk_forward.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'walk_forward.png'} and {RESULTS_DIR / 'compare_all_metrics.json'}")


if __name__ == "__main__":
    main()
