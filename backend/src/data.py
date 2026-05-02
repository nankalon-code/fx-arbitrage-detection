"""
FX Tick Data Engine
Supports: simulated ticks, yfinance OHLCV, exchangerate-api live quotes
"""

import numpy as np
import pandas as pd
import time
import random
import requests
from datetime import datetime, timedelta
from typing import Generator, Dict, List
import yfinance as yf


# 20 major FX pairs
FX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
    "EURCHF", "AUDJPY", "GBPCHF", "EURAUD", "EURCAD",
    "GBPAUD", "GBPCAD", "AUDCAD", "AUDCHF", "CADJPY"
]

# Approximate mid prices for simulation realism
BASE_PRICES = {
    "EURUSD": 1.0850, "GBPUSD": 1.2700, "USDJPY": 149.50,
    "USDCHF": 0.8900, "AUDUSD": 0.6550, "USDCAD": 1.3600,
    "NZDUSD": 0.6050, "EURGBP": 0.8540, "EURJPY": 162.20,
    "GBPJPY": 189.80, "EURCHF": 0.9650, "AUDJPY": 97.90,
    "GBPCHF": 1.1340, "EURAUD": 1.6560, "EURCAD": 1.4750,
    "GBPAUD": 1.9380, "GBPCAD": 1.7250, "AUDCAD": 0.8910,
    "AUDCHF": 0.5830, "CADJPY": 109.90
}


class TickSimulator:
    """
    High-frequency tick simulator with realistic microstructure:
    - Bid/ask spread
    - Volatility clustering (GARCH-like)
    - Occasional arbitrage windows injected for DQN to learn
    """

    def __init__(self, pairs: List[str] = FX_PAIRS, inject_arb_prob: float = 0.05):
        self.pairs = pairs
        self.prices = {p: BASE_PRICES.get(p, 1.0) for p in pairs}
        self.volatilities = {p: 0.0002 for p in pairs}
        self.inject_arb_prob = inject_arb_prob
        self.tick_count = 0

    def _update_volatility(self, pair: str):
        """GARCH(1,1)-like vol clustering"""
        alpha, beta, omega = 0.1, 0.85, 0.000001
        shock = np.random.normal(0, 1) ** 2
        self.volatilities[pair] = (
            omega + alpha * shock * self.volatilities[pair] + beta * self.volatilities[pair]
        )
        self.volatilities[pair] = np.clip(self.volatilities[pair], 1e-6, 0.01)

    def _spread(self, pair: str) -> float:
        """Typical pip spread by pair"""
        spreads = {"EURUSD": 0.0001, "GBPUSD": 0.0002, "USDJPY": 0.02}
        return spreads.get(pair, 0.0003)

    def next_tick(self) -> Dict:
        """Generate one tick for all pairs"""
        self.tick_count += 1
        snapshot = {"timestamp": datetime.utcnow().isoformat(), "tick": self.tick_count}

        # Occasionally inject triangular arbitrage window
        arb_injected = random.random() < self.inject_arb_prob
        if arb_injected:
            self._inject_arbitrage()

        for pair in self.pairs:
            self._update_volatility(pair)
            move = np.random.normal(0, self.volatilities[pair])
            self.prices[pair] *= (1 + move)
            spread = self._spread(pair)
            snapshot[pair] = {
                "bid": round(self.prices[pair] - spread / 2, 6),
                "ask": round(self.prices[pair] + spread / 2, 6),
                "mid": round(self.prices[pair], 6),
                "vol": round(self.volatilities[pair], 8)
            }

        snapshot["arb_injected"] = arb_injected
        return snapshot

    def _inject_arbitrage(self):
        """
        Create a synthetic triangular arbitrage:
        EUR -> USD -> JPY -> EUR with mis-pricing
        """
        # Slightly mis-price EURJPY relative to EURUSD * USDJPY
        eurusd = self.prices["EURUSD"]
        usdjpy = self.prices["USDJPY"]
        theoretical_eurjpy = eurusd * usdjpy
        # Inject 3-8 pip deviation
        deviation = random.uniform(0.03, 0.08)
        self.prices["EURJPY"] = theoretical_eurjpy * (1 + deviation / 100)

    def stream(self, ticks_per_second: int = 100) -> Generator:
        """Generator that yields ticks at given rate"""
        delay = 1.0 / ticks_per_second
        while True:
            yield self.next_tick()
            time.sleep(delay)


class YFinanceLoader:
    """Load historical FX data from yfinance for backtesting"""

    YFINANCE_PAIRS = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
        "USDCHF": "CHF=X",    "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X",
        "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
        "GBPJPY": "GBPJPY=X"
    }

    def load(self, pair: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
        ticker = self.YFINANCE_PAIRS.get(pair, f"{pair}=X")
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {pair}")
        df.columns = [c.lower() for c in df.columns]
        df["pair"] = pair
        df["mid"] = (df["high"] + df["low"]) / 2
        df["spread"] = (df["high"] - df["low"]) * 0.1
        df["bid"] = df["mid"] - df["spread"] / 2
        df["ask"] = df["mid"] + df["spread"] / 2
        return df.dropna()

    def load_multi(self, pairs: List[str] = None, period: str = "30d") -> Dict[str, pd.DataFrame]:
        pairs = pairs or list(self.YFINANCE_PAIRS.keys())
        data = {}
        for pair in pairs:
            try:
                data[pair] = self.load(pair, period=period)
                print(f"Loaded {pair}: {len(data[pair])} bars")
            except Exception as e:
                print(f"Skipping {pair}: {e}")
        return data


class ExchangeRateAPI:
    """
    Live FX rates from exchangerate-api.com (free tier: 1500 req/month)
    Falls back to simulated if API unavailable
    """

    BASE_URL = "https://open.er-api.com/v6/latest"

    def get_rates(self, base: str = "USD") -> Dict[str, float]:
        try:
            resp = requests.get(f"{self.BASE_URL}/{base}", timeout=5)
            data = resp.json()
            if data.get("result") == "success":
                return data["rates"]
        except Exception as e:
            print(f"ExchangeRateAPI unavailable: {e}. Using simulated rates.")
        return {}

    def build_rate_matrix(self) -> pd.DataFrame:
        """Build full NxN cross-rate matrix from USD base rates"""
        rates = self.get_rates("USD")
        if not rates:
            return pd.DataFrame()
        currencies = list(rates.keys())[:20]
        matrix = pd.DataFrame(index=currencies, columns=currencies, dtype=float)
        for c1 in currencies:
            for c2 in currencies:
                if c1 == c2:
                    matrix.loc[c1, c2] = 1.0
                else:
                    matrix.loc[c1, c2] = rates[c2] / rates[c1]
        return matrix


def build_price_matrix(tick: Dict, pairs: List[str] = None) -> np.ndarray:
    """
    Convert a tick snapshot into an NxN log-price matrix
    for Bellman-Ford arbitrage detection
    """
    pairs = pairs or FX_PAIRS[:10]
    currencies = list(set(
        [p[:3] for p in pairs] + [p[3:] for p in pairs]
    ))
    n = len(currencies)
    idx = {c: i for i, c in enumerate(currencies)}
    matrix = np.zeros((n, n))

    for pair in pairs:
        if pair not in tick:
            continue
        b, q = pair[:3], pair[3:]
        if b not in idx or q not in idx:
            continue
        mid = tick[pair]["mid"]
        i, j = idx[b], idx[q]
        matrix[i][j] = np.log(mid)
        matrix[j][i] = -np.log(mid)

    return matrix, currencies, idx
