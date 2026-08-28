"""₩10,000 virtual scalping run: RSI(2) mean-reversion bracket scalp on Upbit BTC/KRW 1-minute bars."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.data import fetch_ohlcv
from src.scalping_backtest import run_scalp_backtest

RESULTS_DIR = Path(__file__).resolve().parent / "results"
STARTING_CAPITAL_KRW = 10_000


def main() -> None:
    df = fetch_ohlcv("upbit", "BTC/KRW", timeframe="1m", since="2026-07-30")
    print(f"Loaded {len(df)} 1-minute candles: {df.index[0]} -> {df.index[-1]}")

    result = run_scalp_backtest(df, starting_capital=STARTING_CAPITAL_KRW)
    m = result.metrics

    print(f"\nStarting capital: {STARTING_CAPITAL_KRW:,}원")
    print(f"Final capital:    {m['final_capital']:,.0f}원")
    print(f"Total return:     {m['total_return_pct']}%")
    print(f"Trades:           {m['n_trades']}  (win rate {m['win_rate_pct']}%)")
    print(f"  take-profit hits: {m['tp_hits']}   stop-loss hits: {m['sl_hits']}   timeouts: {m['timeouts']}")
    print(f"Avg return/trade: {m['avg_return_per_trade_pct']}%")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "scalp_metrics.json", "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    if not result.trades.empty:
        result.trades.to_csv(RESULTS_DIR / "scalp_trades.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(result.equity_curve.index, result.equity_curve.values)
    axes[0].set_title(f"Equity curve — start {STARTING_CAPITAL_KRW:,}원")
    axes[0].set_ylabel("KRW")
    axes[0].tick_params(axis="x", rotation=30)

    if not result.trades.empty:
        axes[1].hist(result.trades["return_pct"], bins=30)
        axes[1].axvline(0, color="black", linewidth=1)
        axes[1].set_title(f"Per-trade return distribution ({len(result.trades)} trades)")
        axes[1].set_xlabel("return % per trade (after both fees)")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scalp_result.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'scalp_result.png'}")


if __name__ == "__main__":
    main()
