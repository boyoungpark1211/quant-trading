"""SMA 50/200 signal -> Binance Futures TESTNET order. Dry-run by default.

Safety by design:
  - src/exchange.py is hardcoded to sandbox mode; this script has no path to a real account.
  - Defaults to leverage=1x. Our own backtest (run_leverage.py) showed this exact strategy
    gets liquidated repeatedly at 3x+ and loses to funding costs even at 2x — so this script
    won't silently default to something we already have evidence is a bad idea.
  - Won't place any order at all unless you pass --live. Without it, it just prints what it
    would have done.

Usage:
  python3 testnet_bot.py             # dry run — prints the signal and intended action only
  python3 testnet_bot.py --live      # actually places the order on TESTNET (fake money)
"""

import argparse

from src.data import fetch_ohlcv
from src.exchange import TestnetKeysMissing, get_testnet_exchange
from src.strategies import sma_crossover

SYMBOL = "BTC/USDT"
MARGIN_USDT = 20  # fake testnet USDT — irrelevant to real risk, kept small out of habit
LEVERAGE = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="place the order on testnet (default: dry run)")
    args = parser.parse_args()

    df = fetch_ohlcv("binance", SYMBOL, timeframe="1d", since="2023-01-01")
    position = sma_crossover(df, fast=50, slow=200)
    current_signal = int(position.iloc[-1])
    last_price = df["close"].iloc[-1]

    print(f"{SYMBOL} last close: {last_price:.2f} ({df.index[-1].date()})")
    print(f"SMA 50/200 signal: {'LONG' if current_signal else 'FLAT'}")

    try:
        exchange = get_testnet_exchange()
    except TestnetKeysMissing as e:
        print(f"\n[no testnet connection] {e}")
        print("Signal computed above; nothing else to report without keys.")
        return

    balance = exchange.fetch_balance()
    usdt = balance.get("USDT", {}).get("free", 0)
    positions = exchange.fetch_positions([SYMBOL])
    open_position = next((p for p in positions if float(p.get("contracts") or 0) != 0), None)

    print(f"\nTestnet USDT balance: {usdt}")
    print(f"Open position: {open_position['contracts'] if open_position else 'none'}")

    want_long = current_signal == 1
    have_long = open_position is not None

    if want_long and not have_long:
        notional = MARGIN_USDT * LEVERAGE
        amount = round(notional / last_price, 5)
        action = f"OPEN long, ~{amount} BTC (~{notional} USDT notional at {LEVERAGE}x)"
    elif have_long and not want_long:
        action = f"CLOSE existing long ({open_position['contracts']} contracts)"
    else:
        action = "no action — already in the desired state"

    print(f"Intended action: {action}")

    if not args.live:
        print("\nDry run — no order placed. Re-run with --live to actually place it on testnet.")
        return

    if want_long and not have_long:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        order = exchange.create_order(SYMBOL, "market", "buy", amount)
        print(f"\nOrder placed on testnet: {order['id']}")
    elif have_long and not want_long:
        order = exchange.create_order(SYMBOL, "market", "sell", abs(float(open_position["contracts"])), params={"reduceOnly": True})
        print(f"\nOrder placed on testnet: {order['id']}")
    else:
        print("\nNothing to do.")


if __name__ == "__main__":
    main()
