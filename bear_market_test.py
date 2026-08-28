"""Isolate known BTC bear/crash windows and check whether each strategy actually defends
capital, rather than just looking good over a mostly-bull 8-year sample.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import buy_and_hold, donchian_breakout, sma_crossover

RESULTS_DIR = Path(__file__).resolve().parent / "results"

WINDOWS = {
    "2018 bear": ("2018-01-01", "2018-12-31"),
    "2020 COVID crash": ("2020-02-01", "2020-04-30"),
    "2022 bear (Luna/FTX)": ("2022-01-01", "2022-12-31"),
}


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2017-01-01")

    strategies = {
        "buy_and_hold": buy_and_hold(df),
        "sma_50_200": sma_crossover(df, 50, 200),
        "donchian_20_10": donchian_breakout(df, entry_lookback=20, exit_lookback=10),
    }

    all_results = {}
    for window_name, (start, end) in WINDOWS.items():
        print(f"\n=== {window_name} ({start} .. {end}) ===")
        window_df = df[(df.index >= start) & (df.index <= end)]
        window_price_change = (window_df["close"].iloc[-1] / window_df["close"].iloc[0] - 1) * 100
        print(f"BTC price change over window: {window_price_change:.1f}%")
        print(f"{'strategy':<16} {'return%':>10} {'max_dd%':>9} {'trades':>7}")

        window_results = {}
        for name, full_position in strategies.items():
            pos_in_window = full_position.reindex(window_df.index)
            result = run_backtest(window_df, pos_in_window)
            window_results[name] = result.metrics
            m = result.metrics
            print(f"{name:<16} {m['total_return_pct']:>10} {m['max_drawdown_pct']:>9} {m['n_trades']:>7}")
        all_results[window_name] = window_results

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "bear_market_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(6 * len(WINDOWS), 5))
    for ax, (window_name, (start, end)) in zip(axes, WINDOWS.items()):
        window_df = df[(df.index >= start) & (df.index <= end)]
        for name, full_position in strategies.items():
            pos_in_window = full_position.reindex(window_df.index)
            result = run_backtest(window_df, pos_in_window)
            ax.plot(result.equity_curve.index, result.equity_curve.values, label=name)
        ax.set_title(window_name)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bear_market_equity.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'bear_market_equity.png'} and {RESULTS_DIR / 'bear_market_metrics.json'}")


if __name__ == "__main__":
    main()
