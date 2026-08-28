# Quant Trading — Strategy Research & Backtest Engine

Crypto strategy research, built to be exchange-agnostic (via [ccxt](https://github.com/ccxt/ccxt)) so the same
backtest code works against Binance, Upbit, Bithumb, etc.

**Scope today: research and backtesting only. No live trading, no exchange API keys, no real capital.**
Turning any strategy here into live execution is a separate, deliberate step — see "Going live" below.

## Structure

```
src/
  data.py               # OHLCV + historical funding-rate fetch, local CSV cache via ccxt (public, no key needed)
  backtest.py            # spot/unleveraged vectorized backtest + metrics
  leverage_backtest.py   # leverage-aware backtest: daily-rebalanced exposure, intrabar liquidation, real funding cost
  strategies.py           # strategy functions: price data in, position signal out
  exchange.py             # Binance Futures TESTNET connection only — hardcoded to sandbox, no path to a real account
run.py                    # spot backtest entry point
run_leverage.py            # leverage-sweep entry point
testnet_bot.py              # SMA signal -> testnet order, dry-run by default (--live to actually place it)
results/                    # metrics/plots from the last run (gitignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the baseline backtest

```bash
source .venv/bin/activate
python3 run.py
```

Fetches BTC/USDT daily candles from Binance's public API, runs an SMA crossover strategy, prints
performance metrics, and writes `results/equity_curve.png` + `results/metrics.json`.

## Adding a strategy

Add a function to `src/strategies.py` with the signature `(df: pd.DataFrame, **params) -> pd.Series`
returning a position series (1 = long, 0 = flat) indexed the same as `df`. Wire it up in `run.py`.

## Leverage sweep

```bash
python3 run_leverage.py
```

Same SMA 50/200 strategy, run at 1x–20x, using **real historical Binance funding-rate data** (not
an assumed constant) and a per-day liquidation check against the day's low. Findings so far: this
strategy loses to funding + daily-rebalance decay by 2x, and gets liquidated repeatedly at 3x+ —
worth rerunning before trusting any leverage level on a different strategy.

## Testnet bot (paper trading, real order flow, zero real money)

```bash
cp .env.example .env   # fill in free keys from https://testnet.binancefuture.com
python3 testnet_bot.py          # dry run — prints the signal + intended action, places nothing
python3 testnet_bot.py --live   # places the order for real, but only on the TESTNET (fake USDT)
```

`src/exchange.py` is hardcoded to `set_sandbox_mode(True)` — there's no flag or config in this repo
that reaches a real Binance account. Defaults to 1x leverage, since the leverage sweep above already
showed this strategy doesn't survive 3x+.

## Going to real capital (not built here, deliberately)

Nothing in this repo can place a real order — `exchange.py` only ever connects to testnet. If a
strategy earns enough confidence from backtesting *and* a real testnet run to justify real money,
wiring it to a live account is a separate, deliberate step you take yourself, with your own API
keys, in an environment you control.
