"""Walk-forward validation for the Donchian candidates that beat the original 20/10 pick in the
full-period sweep. A parameter that only wins on the exact period it was picked from isn't an
upgrade -- it has to hold up split into train/test, same standard applied to everything else.
"""

import json
from pathlib import Path

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import buy_and_hold, donchian_breakout

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLIT_DATE = "2023-01-01"

CANDIDATES = {
    "donchian_20_10 (original)": (20, 10),
    "donchian_25_20 (best full-period)": (25, 20),
    "donchian_40_20": (40, 20),
    "donchian_70_20 (lowest drawdown)": (70, 20),
    "donchian_25_10": (25, 10),
}


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")
    train_df = df[df.index < SPLIT_DATE]
    test_df = df[df.index >= SPLIT_DATE]

    bh_train = run_backtest(train_df, buy_and_hold(train_df)).metrics
    bh_test = run_backtest(test_df, buy_and_hold(test_df)).metrics
    print(f"buy_and_hold        train: sharpe={bh_train['sharpe']:.2f} cagr={bh_train['cagr_pct']}%  "
          f"|  test: sharpe={bh_test['sharpe']:.2f} cagr={bh_test['cagr_pct']}%\n")

    results = {}
    print(f"{'strategy':<32} | {'train Sharpe':>12}{'train CAGR%':>13}{'train MDD%':>12} | "
          f"{'test Sharpe':>12}{'test CAGR%':>12}{'test MDD%':>11}{'test trades':>13}")
    for name, (entry, exit_) in CANDIDATES.items():
        full_position = donchian_breakout(df, entry_lookback=entry, exit_lookback=exit_)
        train_m = run_backtest(train_df, full_position.reindex(train_df.index)).metrics
        test_m = run_backtest(test_df, full_position.reindex(test_df.index)).metrics
        results[name] = {"train": train_m, "test": test_m}
        print(
            f"{name:<32} | {train_m['sharpe']:>12}{train_m['cagr_pct']:>13}{train_m['max_drawdown_pct']:>12} | "
            f"{test_m['sharpe']:>12}{test_m['cagr_pct']:>12}{test_m['max_drawdown_pct']:>11}{test_m['n_trades']:>13}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "donchian_upgrade_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'donchian_upgrade_validation.json'}")


if __name__ == "__main__":
    main()
