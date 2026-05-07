"""
Valuation Engine Module

Provides valuation analysis tools for the Streamlit dashboard:
- RegimeMonteCarloEngine: Monte Carlo simulation for price forecasts
- ScenarioEngine: Scenario-based valuation analysis
- ETFFairValueEngine: Fair value calculations for ETFs
"""

import numpy as np
from typing import Dict, List, Any, Optional
import random
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    """Result container for Monte Carlo simulation."""
    median_terminal: float
    initial_value: float
    median_cagr: float
    expected_terminal: float
    var_95: float
    cvar_95: float
    prob_profit: float
    prob_10pct: float
    prob_20pct: float
    expected_max_drawdown: float
    median_sharpe: float
    percentiles: Dict[int, np.ndarray]
    regime_visit_fractions: Dict[str, float]
    terminal_values: np.ndarray = None


class RegimeMonteCarloEngine:
    """
    Monte Carlo simulation engine for price forecasting.
    Uses regime-aware volatility and drift parameters.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        self._regime_params = self._get_regime_params()
    
    def _get_regime_params(self) -> Dict[str, Dict[str, float]]:
        """Get volatility and drift parameters based on regime."""
        return {
            "Bull": {"drift": 0.08, "volatility": 0.15},
            "Euphoria": {"drift": 0.12, "volatility": 0.22},
            "Mania": {"drift": 0.05, "volatility": 0.28},
            "Neutral": {"drift": 0.03, "volatility": 0.12},
            "Unknown": {"drift": 0.0, "volatility": 0.15},
            "Bear": {"drift": -0.05, "volatility": 0.18},
            "Panic": {"drift": -0.12, "volatility": 0.30},
            "Crash": {"drift": -0.20, "volatility": 0.40},
        }
    
    def run(
        self,
        current_regime: str = "Neutral",
        regime_probs: Optional[Dict[str, float]] = None,
        initial_value: float = 100.0,
        n_simulations: int = 1000,
        horizon_days: int = 252,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for a symbol.
        
        Args:
            current_regime: Current market regime
            regime_probs: Dictionary of regime probabilities
            initial_value: Starting price/value
            n_simulations: Number of simulation paths
            horizon_days: Number of trading days to simulate
            
        Returns:
            MonteCarloResult with simulation statistics
        """
        # Use regime probabilities if provided, otherwise use current regime
        if regime_probs and len(regime_probs) > 0:
            # Weighted average of parameters across regimes
            weighted_drift = 0
            weighted_vol = 0
            total_weight = 0
            for regime, prob in regime_probs.items():
                params = self._regime_params.get(regime, self._regime_params["Neutral"])
                weighted_drift += params["drift"] * prob
                weighted_vol += params["volatility"] * prob
                total_weight += prob
            if total_weight > 0:
                drift = weighted_drift / total_weight / 252
                volatility = weighted_vol / total_weight / np.sqrt(252)
            else:
                params = self._regime_params.get(current_regime, self._regime_params["Neutral"])
                drift = params["drift"] / 252
                volatility = params["volatility"] / np.sqrt(252)
        else:
            params = self._regime_params.get(current_regime, self._regime_params["Neutral"])
            drift = params["drift"] / 252
            volatility = params["volatility"] / np.sqrt(252)
        
        # Generate random price paths using geometric Brownian motion
        dt = 1  # Daily steps
        all_paths = []
        
        for _ in range(n_simulations):
            prices = [initial_value]
            for _ in range(horizon_days):
                shock = np.random.normal(0, 1)
                price_change = drift + volatility * shock
                new_price = prices[-1] * np.exp(price_change)
                prices.append(max(new_price, 0.01))  # Prevent negative prices
            all_paths.append(prices)
        
        # Calculate statistics
        final_prices = [path[-1] for path in all_paths]
        terminal_values = np.array(final_prices)
        
        # Calculate returns
        returns = (terminal_values / initial_value) - 1
        
        # Percentiles over time
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[p] = np.percentile(all_paths, p, axis=0)
        
        # Calculate metrics
        median_terminal = np.median(terminal_values)
        expected_terminal = np.mean(terminal_values)
        
        # CAGR calculation
        median_cagr = (median_terminal / initial_value) ** (252 / horizon_days) - 1
        expected_cagr = (expected_terminal / initial_value) ** (252 / horizon_days) - 1
        
        # VaR and CVaR (95% confidence)
        sorted_returns = np.sort(returns)
        var_95_idx = int(0.05 * n_simulations)
        var_95 = initial_value * (1 + sorted_returns[var_95_idx])
        cvar_95 = initial_value * (1 + np.mean(sorted_returns[:var_95_idx]))
        
        # Probability metrics
        prob_profit = np.mean(returns > 0)
        prob_10pct = np.mean(returns > 0.10)
        prob_20pct = np.mean(returns > 0.20)
        
        # Calculate max drawdown for each path
        max_drawdowns = []
        for path in all_paths:
            peak = path[0]
            max_dd = 0
            for price in path:
                if price > peak:
                    peak = price
                dd = (peak - price) / peak
                if dd > max_dd:
                    max_dd = dd
            max_drawdowns.append(max_dd)
        expected_max_drawdown = np.mean(max_drawdowns)
        
        # Sharpe ratio (using median path)
        median_path = np.median(all_paths, axis=0)
        path_returns = np.diff(median_path) / median_path[:-1]
        path_std = np.std(path_returns)
        median_sharpe = (expected_cagr - 0.02) / path_std if path_std > 0 else 0
        
        # Calculate regime visit fractions (simulated regime transitions)
        # This is a simplified model - in reality would use Markov chain transitions
        regime_visit_fractions = {}
        if regime_probs and len(regime_probs) > 0:
            regime_visit_fractions = regime_probs.copy()
        else:
            # Default to current regime with small probability of transitions
            regime_visit_fractions = {current_regime: 0.85}
            for reg in self._regime_params.keys():
                if reg != current_regime:
                    regime_visit_fractions[reg] = 0.15 / (len(self._regime_params) - 1)
        
        return MonteCarloResult(
            median_terminal=median_terminal,
            initial_value=initial_value,
            median_cagr=median_cagr,
            expected_terminal=expected_terminal,
            var_95=var_95,
            cvar_95=cvar_95,
            prob_profit=prob_profit,
            prob_10pct=prob_10pct,
            prob_20pct=prob_20pct,
            expected_max_drawdown=expected_max_drawdown,
            median_sharpe=median_sharpe,
            percentiles=percentiles,
            regime_visit_fractions=regime_visit_fractions,
            terminal_values=terminal_values,
        )
    
    def get_price_distribution(
        self,
        symbol: str,
        current_price: float,
        days: int = 252
    ) -> List[float]:
        """Get final price distribution after simulation."""
        result = self.run(
            current_regime="Neutral",
            initial_value=current_price,
            n_simulations=1000,
            horizon_days=days
        )
        return sorted([path[-1] for path in result.percentiles[50]])


@dataclass
class ScenarioResult:
    """One scenario returned by ScenarioEngine.run()."""
    name: str
    description: str
    probability: float
    color: str
    risk_level: str
    action: str
    expected_return_1m: float
    expected_return_3m: float
    expected_return_6m: float
    expected_return_12m: float
    portfolio_value_1m: float
    portfolio_value_3m: float
    portfolio_value_6m: float
    portfolio_value_12m: float
    expected_max_drawdown: float
    sharpe_estimate: float


class ScenarioEngine:
    """
    Scenario-based valuation analysis engine.
    Generates bull, base, and bear case scenarios.
    """
    
    def __init__(self, regime: str = "Neutral", confidence: float = 0.5):
        self.regime = regime
        self.confidence = confidence
    
    def generate_scenarios(
        self,
        symbol: str,
        current_price: float,
        days: int = 252
    ) -> Dict[str, Any]:
        """
        Generate bull, base, and bear case scenarios.
        
        Args:
            symbol: Ticker symbol
            current_price: Current price of the asset
            days: Number of trading days for projection
            
        Returns:
            Dictionary with three scenario projections
        """
        # Define scenario parameters based on regime
        scenario_params = {
            "Bull": {
                "bull": {"drift": 0.15, "volatility": 0.18},
                "base": {"drift": 0.06, "volatility": 0.12},
                "bear": {"drift": -0.05, "volatility": 0.20},
            },
            "Euphoria": {
                "bull": {"drift": 0.20, "volatility": 0.25},
                "base": {"drift": 0.08, "volatility": 0.18},
                "bear": {"drift": -0.08, "volatility": 0.30},
            },
            "Mania": {
                "bull": {"drift": 0.12, "volatility": 0.28},
                "base": {"drift": 0.02, "volatility": 0.22},
                "bear": {"drift": -0.12, "volatility": 0.35},
            },
            "Neutral": {
                "bull": {"drift": 0.10, "volatility": 0.14},
                "base": {"drift": 0.04, "volatility": 0.10},
                "bear": {"drift": -0.04, "volatility": 0.16},
            },
            "Bear": {
                "bull": {"drift": 0.05, "volatility": 0.20},
                "base": {"drift": -0.02, "volatility": 0.18},
                "bear": {"drift": -0.10, "volatility": 0.25},
            },
            "Panic": {
                "bull": {"drift": 0.02, "volatility": 0.28},
                "base": {"drift": -0.08, "volatility": 0.28},
                "bear": {"drift": -0.18, "volatility": 0.40},
            },
            "Crash": {
                "bull": {"drift": 0.0, "volatility": 0.30},
                "base": {"drift": -0.15, "volatility": 0.35},
                "bear": {"drift": -0.25, "volatility": 0.50},
            },
        }
        
        params = scenario_params.get(self.regime, scenario_params["Neutral"])
        
        scenarios = {}
        for scenario_name, scenario_params in params.items():
            drift = scenario_params["drift"] / 252
            volatility = scenario_params["volatility"] / np.sqrt(252)
            
            # Simulate price path
            prices = [current_price]
            for _ in range(days):
                shock = np.random.normal(0, 1)
                price_change = drift + volatility * shock
                new_price = prices[-1] * np.exp(price_change)
                prices.append(max(new_price, 0.01))
            
            scenarios[scenario_name] = {
                "final_price": prices[-1],
                "return": (prices[-1] / current_price - 1) * 100,
                "path": prices,
            }
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "days": days,
            "scenarios": scenarios,
        }

    def run(
        self,
        current_regime: str = "Neutral",
        regime_probs: Optional[Dict[str, float]] = None,
        portfolio_value: float = 100_000.0,
    ) -> List["ScenarioResult"]:
        """
        Run bull / base / bear scenario analysis for a portfolio.

        Returns a list of three ScenarioResult objects ordered by
        probability (highest first).
        """
        regime_probs = regime_probs or {}

        # ── Regime-aware annualised return assumptions ─────────────────────────
        _params: Dict[str, Dict] = {
            "Bull":     {"bull": (0.28, 0.15), "base": (0.12, 0.11), "bear": (-0.08, 0.18)},
            "Euphoria": {"bull": (0.35, 0.22), "base": (0.14, 0.16), "bear": (-0.12, 0.25)},
            "Mania":    {"bull": (0.20, 0.28), "base": (0.05, 0.22), "bear": (-0.18, 0.32)},
            "Neutral":  {"bull": (0.18, 0.14), "base": (0.07, 0.10), "bear": (-0.06, 0.15)},
            "Unknown":  {"bull": (0.15, 0.16), "base": (0.04, 0.12), "bear": (-0.08, 0.18)},
            "Bear":     {"bull": (0.10, 0.20), "base": (-0.04, 0.17), "bear": (-0.18, 0.24)},
            "Panic":    {"bull": (0.05, 0.28), "base": (-0.12, 0.28), "bear": (-0.28, 0.40)},
            "Crash":    {"bull": (0.02, 0.30), "base": (-0.20, 0.35), "bear": (-0.40, 0.50)},
        }
        p = _params.get(current_regime, _params["Neutral"])

        # ── Regime-aware probabilities ─────────────────────────────────────────
        _prob_map: Dict[str, Dict[str, float]] = {
            "Bull":     {"bull": 0.50, "base": 0.35, "bear": 0.15},
            "Euphoria": {"bull": 0.45, "base": 0.35, "bear": 0.20},
            "Mania":    {"bull": 0.35, "base": 0.35, "bear": 0.30},
            "Neutral":  {"bull": 0.30, "base": 0.45, "bear": 0.25},
            "Unknown":  {"bull": 0.28, "base": 0.44, "bear": 0.28},
            "Bear":     {"bull": 0.20, "base": 0.35, "bear": 0.45},
            "Panic":    {"bull": 0.15, "base": 0.30, "bear": 0.55},
            "Crash":    {"bull": 0.10, "base": 0.25, "bear": 0.65},
        }
        probs = _prob_map.get(current_regime, _prob_map["Neutral"])

        def _horizon_ret(ann_ret: float, days: int) -> float:
            return (1 + ann_ret) ** (days / 252) - 1

        def _max_dd(ann_ret: float, ann_vol: float) -> float:
            """Approximate expected max drawdown from GBM parameters."""
            return -abs(max(ann_vol * 0.8, -ann_ret * 0.6 + ann_vol * 0.4))

        def _sharpe(ann_ret: float, ann_vol: float) -> float:
            return (ann_ret - 0.045) / ann_vol if ann_vol > 0 else 0.0

        _scenario_meta = {
            "bull": {
                "name": "Bull Case",
                "color": "#00c851",
                "risk_level": "Low",
                "action": "Increase",
                "description": (
                    f"Favourable conditions persist in the {current_regime} regime. "
                    "Risk-on assets outperform; momentum remains constructive."
                ),
            },
            "base": {
                "name": "Base Case",
                "color": "#9e9e9e",
                "risk_level": "Moderate",
                "action": "Hold",
                "description": (
                    f"Consensus outcome for a {current_regime} regime. "
                    "Returns in line with historical averages; no major regime shift."
                ),
            },
            "bear": {
                "name": "Bear Case",
                "color": "#ff4444",
                "risk_level": "High" if current_regime in ("Panic", "Crash") else "Elevated",
                "action": "Reduce" if current_regime in ("Bear", "Panic", "Crash") else "Monitor / Trim",
                "description": (
                    f"Adverse outcome in the {current_regime} regime. "
                    "Risk-off rotation, liquidity stress, or macro shock materialises."
                ),
            },
        }

        results: List[ScenarioResult] = []
        for key in ("bull", "base", "bear"):
            ann_ret, ann_vol = p[key]
            meta = _scenario_meta[key]
            r1m  = _horizon_ret(ann_ret, 21)
            r3m  = _horizon_ret(ann_ret, 63)
            r6m  = _horizon_ret(ann_ret, 126)
            r12m = _horizon_ret(ann_ret, 252)
            results.append(ScenarioResult(
                name=meta["name"],
                description=meta["description"],
                probability=probs[key],
                color=meta["color"],
                risk_level=meta["risk_level"],
                action=meta["action"],
                expected_return_1m=r1m,
                expected_return_3m=r3m,
                expected_return_6m=r6m,
                expected_return_12m=r12m,
                portfolio_value_1m=portfolio_value * (1 + r1m),
                portfolio_value_3m=portfolio_value * (1 + r3m),
                portfolio_value_6m=portfolio_value * (1 + r6m),
                portfolio_value_12m=portfolio_value * (1 + r12m),
                expected_max_drawdown=_max_dd(ann_ret, ann_vol),
                sharpe_estimate=_sharpe(ann_ret, ann_vol),
            ))

        return sorted(results, key=lambda s: s.probability, reverse=True)

    def expected_value(
        self,
        scenarios: List["ScenarioResult"],
        initial_value: float,
    ) -> Dict[str, float]:
        """
        Probability-weighted expected portfolio value across all scenarios.

        Returns a dict with keys ev_1m / ret_1m, ev_3m / ret_3m,
        ev_6m / ret_6m, ev_12m / ret_12m.
        """
        def _ev(attr: str) -> float:
            return sum(getattr(s, attr) * s.probability for s in scenarios)

        ev_1m  = _ev("portfolio_value_1m")
        ev_3m  = _ev("portfolio_value_3m")
        ev_6m  = _ev("portfolio_value_6m")
        ev_12m = _ev("portfolio_value_12m")
        iv = initial_value if initial_value > 0 else 1.0
        return {
            "ev_1m":  ev_1m,  "ret_1m":  ev_1m  / iv - 1,
            "ev_3m":  ev_3m,  "ret_3m":  ev_3m  / iv - 1,
            "ev_6m":  ev_6m,  "ret_6m":  ev_6m  / iv - 1,
            "ev_12m": ev_12m, "ret_12m": ev_12m / iv - 1,
        }

    def calculate_fair_value(
        self,
        symbol: str,
        current_price: float,
        growth_rate: float = 0.05,
        discount_rate: float = 0.08,
        terminal_growth: float = 0.02,
        years: int = 10
    ) -> Dict[str, float]:
        """
        Calculate fair value using discounted cash flow (DCF) model.
        
        Args:
            symbol: Ticker symbol
            current_price: Current price
            growth_rate: Expected annual growth rate
            discount_rate: Discount rate (WACC)
            terminal_growth: Terminal growth rate
            years: Projection period
            
        Returns:
            Dictionary with fair value calculations
        """
        cash_flows = []
        cf = current_price * 0.05  # Assume 5% free cash flow yield
        
        for year in range(1, years + 1):
            cf *= (1 + growth_rate)
            cash_flows.append(cf)
        
        # Calculate present value of cash flows
        pv_cash_flows = sum(
            cf / ((1 + discount_rate) ** year)
            for year, cf in enumerate(cash_flows, 1)
        )
        
        # Terminal value
        terminal_value = (cash_flows[-1] * (1 + terminal_growth)) / (
            discount_rate - terminal_growth
        )
        pv_terminal = terminal_value / ((1 + discount_rate) ** years)
        
        fair_value = pv_cash_flows + pv_terminal
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "fair_value": fair_value,
            "upside": (fair_value / current_price - 1) * 100,
            "discount_rate": discount_rate,
            "growth_rate": growth_rate,
            "terminal_growth": terminal_growth,
        }


@dataclass
class FairValueResult:
    """One instrument's fair-value estimate returned by ETFFairValueEngine.estimate_all()."""
    current_price: float
    fair_value: float
    upside_pct: float
    signal: str
    signal_color: str
    confidence: str
    model: str
    key_metric: str
    regime_note: str


class ETFFairValueEngine:
    """
    Fair value engine for ETF analysis.
    Calculates premium/discount to NAV and identifies trading opportunities.
    """
    
    def __init__(self, risk_free_rate: float = 0.045):
        self.etf_cache = {}
        self.risk_free_rate = risk_free_rate

    def estimate_all(
        self,
        prices: Dict[str, float],
        yield_10y: float = 0.043,
        real_yield: float = 0.019,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, "FairValueResult"]:
        """
        Estimate fair value for each symbol in *prices* using simple
        regime-aware models.  Returns a dict {symbol: FairValueResult}.
        """
        extra_data = extra_data or {}
        rfr = self.risk_free_rate or yield_10y or 0.045
        results: Dict[str, FairValueResult] = {}

        # Per-instrument model parameters (earnings yield, growth, etc.)
        _meta: Dict[str, Dict] = {
            "SPY":  {"model": "Earnings Yield", "eps_yield": 0.046, "growth": 0.07,
                     "key_metric": f"Fwd E/Y {0.046:.1%}", "sector": "US Large Cap"},
            "QQQ":  {"model": "Growth DCF",     "eps_yield": 0.032, "growth": 0.12,
                     "key_metric": f"Fwd E/Y {0.032:.1%}", "sector": "US Tech"},
            "GLD":  {"model": "Real Yield",     "eps_yield": 0.0,   "growth": 0.0,
                     "key_metric": f"Real Yield {real_yield:.2%}", "sector": "Commodity"},
            "TLT":  {"model": "Duration",       "eps_yield": yield_10y, "growth": 0.0,
                     "key_metric": f"10Y {yield_10y:.2%}", "sector": "Fixed Income"},
            "XLE":  {"model": "FCF Yield",      "eps_yield": 0.065, "growth": 0.04,
                     "key_metric": f"FCF Yield 6.5%", "sector": "Energy"},
            "XLK":  {"model": "Growth DCF",     "eps_yield": 0.035, "growth": 0.11,
                     "key_metric": f"Fwd E/Y {0.035:.1%}", "sector": "Technology"},
            "XLF":  {"model": "P/B Model",      "eps_yield": 0.072, "growth": 0.05,
                     "key_metric": f"Fwd E/Y {0.072:.1%}", "sector": "Financials"},
            "XLV":  {"model": "Earnings Yield", "eps_yield": 0.052, "growth": 0.06,
                     "key_metric": f"Fwd E/Y {0.052:.1%}", "sector": "Healthcare"},
        }

        for sym, price in prices.items():
            if not price or price <= 0:
                continue
            m = _meta.get(sym, {
                "model": "Earnings Yield",
                "eps_yield": 0.05,
                "growth": 0.05,
                "key_metric": "Fwd E/Y 5.0%",
                "sector": "Equity",
            })

            eps_yield = m["eps_yield"]
            growth    = m["growth"]

            # Simple Gordon Growth / earnings-yield fair value
            if sym == "GLD":
                # Gold inversely tracks real yield
                fv = price * (1 + max(0.0, 0.02 - real_yield) * 5)
            elif sym == "TLT":
                # Long-duration bond: duration ~17, fair value moves with rate changes
                fair_yield = yield_10y * 1.05
                fv = price * (yield_10y / max(fair_yield, 0.001))
            else:
                required = rfr + 0.03  # equity risk premium ~3%
                if growth >= required:
                    growth = required - 0.005
                fv = price * (eps_yield + growth) / max(required, 0.001)

            upside = (fv / price - 1) * 100 if price > 0 else 0.0

            if upside > 10:
                signal, color = "Undervalued", "#00c851"
            elif upside > 3:
                signal, color = "Mildly Cheap", "#69f0ae"
            elif upside > -3:
                signal, color = "Fair Value", "#9e9e9e"
            elif upside > -10:
                signal, color = "Mildly Rich", "#ffbb33"
            else:
                signal, color = "Overvalued", "#ff4444"

            confidence = "High" if abs(upside) > 15 else ("Medium" if abs(upside) > 5 else "Low")
            regime_note = (
                f"Fair value derived via {m['model']} · rfr {rfr:.2%} · "
                f"growth {growth:.1%} · real yield {real_yield:.2%}"
            )

            results[sym] = FairValueResult(
                current_price=round(price, 2),
                fair_value=round(fv, 2),
                upside_pct=round(upside, 2),
                signal=signal,
                signal_color=color,
                confidence=confidence,
                model=m["model"],
                key_metric=m["key_metric"],
                regime_note=regime_note,
            )

        return results

    def analyze_etf(
        self,
        symbol: str,
        current_price: float,
        nav: Optional[float] = None,
        holdings: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze ETF for fair value and trading signals.
        
        Args:
            symbol: ETF ticker symbol
            current_price: Current trading price
            nav: Net Asset Value (if available)
            holdings: List of holdings with weights and values
            
        Returns:
            Dictionary with valuation analysis
        """
        # If NAV not provided, estimate from holdings or use current price
        if nav is None:
            if holdings:
                nav = sum(h.get("value", 0) for h in holdings)
            else:
                nav = current_price
        
        premium_discount = ((current_price / nav) - 1) * 100 if nav > 0 else 0
        
        # Determine fair value range based on premium/discount history
        fair_value_low = nav * 0.98
        fair_value_high = nav * 1.02
        
        # Trading signal based on premium/discount
        if premium_discount > 2:
            signal = "OVERVALUED"
            action = "SELL"
        elif premium_discount < -2:
            signal = "UNDERVALUED"
            action = "BUY"
        else:
            signal = "FAIR VALUE"
            action = "HOLD"
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "nav": nav,
            "premium_discount": premium_discount,
            "fair_value_low": fair_value_low,
            "fair_value_high": fair_value_high,
            "signal": signal,
            "action": action,
            "holdings": holdings or [],
        }
    
    def compare_etfs(
        self,
        etf_list: List[Dict]
    ) -> pd.DataFrame:
        """
        Compare multiple ETFs for relative value.
        
        Args:
            etf_list: List of ETF dictionaries with symbol, price, nav
            
        Returns:
            DataFrame with comparison metrics
        """
        import pandas as pd
        
        results = []
        for etf in etf_list:
            analysis = self.analyze_etf(
                etf.get("symbol"),
                etf.get("price", 0),
                etf.get("nav")
            )
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def get_sector_exposure(
        self,
        holdings: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate sector exposure from ETF holdings.
        
        Args:
            holdings: List of holdings with sector information
            
        Returns:
            Dictionary mapping sectors to weights
        """
        sector_weights = {}
        total_value = sum(h.get("value", 0) for h in holdings)
        
        if total_value == 0:
            return {}
        
        for holding in holdings:
            sector = holding.get("sector", "Unknown")
            value = holding.get("value", 0)
            weight = (value / total_value) * 100
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        return sector_weights


# Helper function for creating engines based on regime
def create_valuation_engines(regime: str = "Neutral", confidence: float = 0.5):
    """
    Factory function to create all valuation engines.
    
    Args:
        regime: Current market regime
        confidence: Regime confidence level
        
    Returns:
        Tuple of (RegimeMonteCarloEngine, ScenarioEngine, ETFFairValueEngine)
    """
    mc_engine = RegimeMonteCarloEngine(regime, confidence)
    scenario_engine = ScenarioEngine(regime, confidence)
    fair_value_engine = ETFFairValueEngine()
    
    return mc_engine, scenario_engine, fair_value_engine