"""
Realistic FX Transaction Cost Model

Models the REAL costs of executing FX arbitrage:
- Variable bid/ask spreads by pair and market session
- Slippage from market impact
- Execution latency (price moves while your order travels)
- Partial fill probability
- Brokerage/ECN fees

This exists to prove that even "detected" arbitrage rarely survives
real-world execution costs — which is the entire point.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of attempting to execute an arbitrage cycle"""
    gross_profit_bps: float     # Before any costs
    spread_cost_bps: float      # Bid-ask spread paid
    slippage_bps: float         # Market impact
    latency_cost_bps: float     # Price moved during execution
    fee_bps: float              # Broker/ECN fees
    net_profit_bps: float       # What you actually keep
    fill_rate: float            # Fraction of order filled (0-1)
    survived: bool              # Would this trade be profitable after costs?
    legs: int
    path: List[str]
    breakdown: Dict[str, float]


# Typical institutional spreads by pair (in basis points of mid price)
# These are BEST CASE — retail spreads are 2-5x wider
PAIR_SPREADS_BPS = {
    "EURUSD": 0.8,   # Most liquid pair
    "GBPUSD": 1.2,
    "USDJPY": 0.9,
    "USDCHF": 1.5,
    "AUDUSD": 1.3,
    "USDCAD": 1.4,
    "NZDUSD": 2.0,
    "EURGBP": 1.1,
    "EURJPY": 1.5,
    "GBPJPY": 2.5,   # Cross pairs are wider
    "EURCHF": 1.8,
    "AUDJPY": 2.8,
    "GBPCHF": 3.0,
    "EURAUD": 2.5,
    "EURCAD": 2.2,
    "GBPAUD": 3.5,
    "GBPCAD": 3.2,
    "AUDCAD": 3.0,
    "AUDCHF": 3.5,
    "CADJPY": 3.0,
}

# Session multipliers — spreads widen outside major sessions
SESSION_SPREAD_MULTIPLIER = {
    "london_ny_overlap": 1.0,   # 13:00-17:00 UTC — tightest
    "london":           1.2,    # 08:00-16:00 UTC
    "new_york":         1.3,    # 13:00-21:00 UTC
    "tokyo":            1.8,    # 00:00-09:00 UTC
    "sydney":           2.2,    # 21:00-06:00 UTC
    "off_hours":        3.0,    # Weekends, holidays
}


class TransactionCostModel:
    """
    Models realistic execution costs for FX arbitrage.

    The key insight: triangular arbitrage typically yields 1-5 bps gross
    in simulated conditions. After real costs, most of that evaporates.
    """

    def __init__(
        self,
        notional_usd: float = 1_000_000,
        ecn_fee_bps: float = 0.3,
        latency_ms: float = 50.0,
        volatility_scale: float = 1.0,
    ):
        """
        Args:
            notional_usd: Trade size in USD (affects slippage)
            ecn_fee_bps: Broker/ECN fee per leg in basis points
            latency_ms: Round-trip execution latency in milliseconds
            volatility_scale: Multiplier on vol-based costs (1.0 = normal)
        """
        self.notional_usd = notional_usd
        self.ecn_fee_bps = ecn_fee_bps
        self.latency_ms = latency_ms
        self.volatility_scale = volatility_scale

    def estimate_spread_cost(
        self,
        path: List[str],
        session: str = "london_ny_overlap"
    ) -> float:
        """
        Total spread cost for executing all legs of an arb cycle.
        You cross the spread on EVERY leg.
        """
        multiplier = SESSION_SPREAD_MULTIPLIER.get(session, 1.5)
        total = 0.0
        for i in range(len(path) - 1):
            pair = self._path_to_pair(path[i], path[i + 1])
            base_spread = PAIR_SPREADS_BPS.get(pair, 3.0)
            total += base_spread * multiplier
        return total

    def estimate_slippage(
        self,
        path: List[str],
        volatilities: Dict[str, float] = None
    ) -> float:
        """
        Market impact from your order moving the price.

        Uses square-root market impact model:
        slippage ∝ σ × √(Q / ADV)

        For $1M notional on major pairs (ADV ~$100B), this is tiny.
        For crosses or during low liquidity, it matters.
        """
        legs = len(path) - 1
        if legs == 0:
            return 0.0

        # Average daily volume estimates (USD billions)
        adv = {
            "EUR": 800, "USD": 1000, "JPY": 400, "GBP": 350,
            "CHF": 100, "AUD": 150, "CAD": 120, "NZD": 40,
        }

        total_slippage = 0.0
        for i in range(legs):
            ccy = path[i]
            daily_vol = adv.get(ccy, 50) * 1e9
            # Square-root impact model
            participation = self.notional_usd / daily_vol
            vol_bps = 10.0  # ~10 bps daily vol for major FX
            if volatilities:
                pair = self._path_to_pair(path[i], path[i + 1])
                if pair in volatilities:
                    vol_bps = volatilities[pair] * 10000

            impact = vol_bps * np.sqrt(participation) * self.volatility_scale
            total_slippage += impact

        return total_slippage

    def estimate_latency_cost(
        self,
        path: List[str],
        avg_vol_bps: float = 10.0
    ) -> float:
        """
        Price moves WHILE your order is in flight.

        For 50ms latency with 10bps daily vol:
        Expected move ≈ σ_daily × √(Δt / 1day)
                      ≈ 10bps × √(50ms / 86400000ms)
                      ≈ 0.0076 bps per leg

        Seems tiny, but over 3 legs it compounds, and in volatile
        conditions it can be 10x higher.
        """
        legs = len(path) - 1
        dt_fraction = (self.latency_ms / 1000) / 86400  # fraction of a day
        cost_per_leg = avg_vol_bps * np.sqrt(dt_fraction) * self.volatility_scale
        return cost_per_leg * legs

    def estimate_fill_probability(
        self,
        path: List[str],
        profit_bps: float
    ) -> float:
        """
        Probability that ALL legs get filled at quoted prices.

        In reality, by the time leg 2 fills, the price for leg 3
        may have moved. Tighter arb windows = lower fill probability.

        Model: P(fill_all) = P(fill_one)^n_legs
        P(fill_one) depends on how much edge you have vs the spread.
        """
        legs = len(path) - 1
        if legs == 0:
            return 0.0

        # Edge ratio: how much profit vs spread per leg
        avg_spread = self.estimate_spread_cost(path) / max(legs, 1)
        edge_ratio = profit_bps / max(avg_spread * legs, 0.01)

        # Sigmoid-like fill probability
        p_single = min(0.95, max(0.1, 1 / (1 + np.exp(-2 * (edge_ratio - 1)))))
        return p_single ** legs

    def evaluate_opportunity(
        self,
        path: List[str],
        gross_profit_bps: float,
        session: str = "london_ny_overlap",
        volatilities: Dict[str, float] = None,
    ) -> ExecutionResult:
        """
        Full cost analysis for a detected arbitrage opportunity.

        Returns whether this trade would survive real-world execution.
        """
        legs = len(path) - 1

        spread = self.estimate_spread_cost(path, session)
        slippage = self.estimate_slippage(path, volatilities)
        latency = self.estimate_latency_cost(path)
        fees = self.ecn_fee_bps * legs

        total_cost = spread + slippage + latency + fees
        net = gross_profit_bps - total_cost
        fill_rate = self.estimate_fill_probability(path, gross_profit_bps)

        # Expected value accounting for partial fills
        expected_net = net * fill_rate

        return ExecutionResult(
            gross_profit_bps=round(gross_profit_bps, 4),
            spread_cost_bps=round(spread, 4),
            slippage_bps=round(slippage, 4),
            latency_cost_bps=round(latency, 4),
            fee_bps=round(fees, 4),
            net_profit_bps=round(net, 4),
            fill_rate=round(fill_rate, 4),
            survived=expected_net > 0,
            legs=legs,
            path=path,
            breakdown={
                "gross_profit": round(gross_profit_bps, 4),
                "spread_cost": round(spread, 4),
                "slippage": round(slippage, 4),
                "latency_cost": round(latency, 4),
                "ecn_fees": round(fees, 4),
                "total_cost": round(total_cost, 4),
                "net_profit": round(net, 4),
                "fill_probability": round(fill_rate, 4),
                "expected_value": round(expected_net, 4),
            }
        )

    def _path_to_pair(self, base: str, quote: str) -> str:
        """Convert currency pair to standard notation"""
        pair = f"{base}{quote}"
        if pair in PAIR_SPREADS_BPS:
            return pair
        reverse = f"{quote}{base}"
        if reverse in PAIR_SPREADS_BPS:
            return reverse
        return pair


def cost_analysis_summary(results: List[ExecutionResult]) -> Dict:
    """Generate summary statistics from a batch of evaluated opportunities"""
    if not results:
        return {"total": 0, "survived": 0, "killed": 0}

    survived = [r for r in results if r.survived]
    killed = [r for r in results if not r.survived]

    gross_profits = [r.gross_profit_bps for r in results]
    net_profits = [r.net_profit_bps for r in results]
    spread_costs = [r.spread_cost_bps for r in results]

    return {
        "total_opportunities": len(results),
        "survived_after_costs": len(survived),
        "killed_by_costs": len(killed),
        "survival_rate_pct": round(len(survived) / len(results) * 100, 1),
        "avg_gross_profit_bps": round(np.mean(gross_profits), 2),
        "avg_net_profit_bps": round(np.mean(net_profits), 2),
        "avg_spread_cost_bps": round(np.mean(spread_costs), 2),
        "avg_fill_rate": round(np.mean([r.fill_rate for r in results]), 3),
        "max_gross_bps": round(max(gross_profits), 2),
        "max_net_bps": round(max(net_profits), 2) if net_profits else 0,
        "pct_profit_lost_to_costs": round(
            (1 - np.mean(net_profits) / max(np.mean(gross_profits), 0.001)) * 100, 1
        ),
    }
