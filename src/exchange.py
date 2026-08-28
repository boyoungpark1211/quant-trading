"""Binance Futures TESTNET connection only.

This module is hardcoded to sandbox mode — there is no code path in this file that can reach
a real Binance account, no matter what flags are passed. Going from testnet to mainnet is a
deliberate decision that means writing new code, not flipping a switch here.
"""

import os

import ccxt
from dotenv import load_dotenv

load_dotenv()


class TestnetKeysMissing(RuntimeError):
    pass


def get_testnet_exchange() -> ccxt.binanceusdm:
    """Returns a ccxt Binance USDM-futures client pointed at the TESTNET, using keys from .env.

    Raises TestnetKeysMissing with setup instructions if BINANCE_TESTNET_API_KEY/SECRET aren't set.
    """
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET")
    if not api_key or not api_secret:
        raise TestnetKeysMissing(
            "No testnet keys found. Get free ones at https://testnet.binancefuture.com, "
            "then copy .env.example to .env and fill them in."
        )

    exchange = ccxt.binanceusdm({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
    exchange.set_sandbox_mode(True)
    return exchange
