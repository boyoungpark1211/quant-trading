"""Generalization check: does the Donchian 25/10 edge validated on BTC/USDT (see
validate_donchian_upgrade.py -- test Sharpe 1.46, CAGR 46%, max DD -25.6%) transfer to a
different asset, or is it BTC-specific curve-fitting we didn't notice because we only ever
tested one symbol?

Deliberately uses the EXACT SAME params (entry_lookback=25, exit_lookback=10) already chosen
from the BTC sweep -- no re-fitting to ETH. Re-optimizing per-asset would just be a new
overfitting risk and wouldn't answer the actual question (does the BTC-derived rule transfer?).
Same walk-forward split as everywhere else in this repo (train: 2018 .. 2023-01-01, test:
2023-01-01 onward) so the comparison is apples-to-apples with the BTC results already on file.
"""

import json
from pathlib import Path

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import buy_and_hold, donchian_breakout

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLIT_DATE = "2023-01-01"
ENTRY, EXIT = 25, 10


def report(label: str, df, position) -> dict:
    result = run_backtest(df, position.reindex(df.index))
    m = result.metrics
    print(
        f"{label:<10} sharpe={m['sharpe']:>6} cagr%={m['cagr_pct']:>7} "
        f"max_dd%={m['max_drawdown_pct']:>7} trades={m['n_trades']:>4} n_days={m['n_days']}"
    )
    return m


def main() -> None:
    eth = fetch_ohlcv("binance", "ETH/USDT", timeframe="1d", since="2018-01-01")
    btc = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")

    eth_position = donchian_breakout(eth, entry_lookback=ENTRY, exit_lookback=EXIT)
    btc_position = donchian_breakout(btc, entry_lookback=ENTRY, exit_lookback=EXIT)

    eth_train, eth_test = eth[eth.index < SPLIT_DATE], eth[eth.index >= SPLIT_DATE]
    btc_train, btc_test = btc[btc.index < SPLIT_DATE], btc[btc.index >= SPLIT_DATE]

    results = {}

    print(f"=== donchian_{ENTRY}_{EXIT}, ETH/USDT vs BTC/USDT (same params, no re-fitting) ===\n")

    print("-- Full period 2018-2026 --")
    results["eth_full"] = report("ETH", eth, eth_position)
    results["btc_full"] = report("BTC", btc, btc_position)
    results["eth_full_bh"] = report("ETH b&h", eth, buy_and_hold(eth))
    results["btc_full_bh"] = report("BTC b&h", btc, buy_and_hold(btc))

    print(f"\n-- Train (2018 .. {SPLIT_DATE}) --")
    results["eth_train"] = report("ETH", eth_train, eth_position)
    results["btc_train"] = report("BTC", btc_train, btc_position)
    results["eth_train_bh"] = report("ETH b&h", eth_train, buy_and_hold(eth_train))
    results["btc_train_bh"] = report("BTC b&h", btc_train, buy_and_hold(btc_train))

    print(f"\n-- Test / out-of-sample ({SPLIT_DATE} .. 2026) --")
    results["eth_test"] = report("ETH", eth_test, eth_position)
    results["btc_test"] = report("BTC", btc_test, btc_position)
    results["eth_test_bh"] = report("ETH b&h", eth_test, buy_and_hold(eth_test))
    results["btc_test_bh"] = report("BTC b&h", btc_test, buy_and_hold(btc_test))

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "eth_generalization.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'eth_generalization.json'}")


if __name__ == "__main__":
    main()
