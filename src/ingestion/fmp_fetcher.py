# Path: src/fetchers/fmp_fetcher.py
"""EU/Asia market data fetcher — FMP Ultimate stable/ routes.

Phase-0 smoke-test (2026-05-30) confirmed FMP stable/ routes work for EU/Asia
symbols: historical prices, quote, ratios-ttm, news/stock, insider-trading/search,
upgrades-downgrades-consensus-bulk, analyst-estimates, price-target-consensus.

PATCH v2.2-global (2026-06):
Previously this fetcher only retrieved historical prices (momentum + volume).
FMP Ultimate covers all factor sources globally — the "US-only" limitation was
based on tests against the retired /api/v3/ routes, not stable/.

Factors now fetched for ALL markets:
  - momentum_long_score:       historical-price-eod/full  (unchanged)
  - volume_attention_score:    historical-price-eod/full  (unchanged)
  - insider_conviction_score:  insider-trading/search     (NEW — MAR Art.19)
  - insider_breadth_score:     insider-trading/search     (NEW)
  - news_sentiment_score:      news/stock                 (NEW)
  - news_buzz_score:           news/stock                 (NEW)
  - analyst_consensus_score:   bulk index lookup          (NEW)
  - analyst_revision_score:    analyst-estimates          (NEW)
  - quality_piotroski_score:   ratios-ttm                 (unchanged path)
  - price_target_upside_score: price-target-consensus     (unchanged path)

Structurally absent (no data source exists globally):
  - congress_score: US STOCK Act / S3 Stock Watcher only — always 0.0
  - transcript_tone_score: FMP earning-call-transcript-latest US-only — always 0.0

Insider data coverage (critical for EU/Asia):
  FMP insider-trading/search is SEC Form 4 data — US STOCK Act disclosures only.
  EU/Asia companies file under MAR Art.19 (EU) or local regimes, which are NOT
  indexed by FMP. For non-US tickers (those with an exchange suffix like .AS,
  .PA, .KS, .T), insider_conviction_score and insider_breadth_score are forced
  to None (excluded from the StrategyEngine denominator) rather than 0.0
  (which would penalise them as genuine "no insider buying" signal).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.core.fetchers_base import BaseMarketFetcher, MarketEnum, TickerEntry
from src.ingestion.v3_shadow import v3_shadow_enabled as _v3_shadow_enabled

logger = logging.getLogger(__name__)

_VOL_BASELINE_BARS = 90
_VOL_BASELINE_SKIP = 5
_VOL_MAX_SPIKE = 20.0
_MIN_BARS_MOMENTUM = 252
_PRICE_LIMIT = 280   # 13 months of trading days


class FMPFetcher(BaseMarketFetcher):
    """Global market data fetcher using FMP Ultimate stable/ routes.

    Covers US, EU (XETRA, LSE, Euronext, SIX) and Asia (TSE, KRX, HKEX, NSE).
    All factor endpoints confirmed live on FMP Ultimate for non-US symbols.

    congress_score is the ONLY factor that cannot be fetched outside the US —
    it relies on STOCK Act disclosures via S3 Stock Watcher (US-only feeds).
    All other factors (insider, news, analyst, momentum, quality) are available
    globally via FMP Ultimate.
    """

    def __init__(
        self,
        api_key: str = "",
        market: MarketEnum = MarketEnum.EUROPE,
        bulk_consensus_idx: Optional[dict] = None,
        ambiguous_bases: Optional[set] = None,
    ) -> None:
        self._api_key = api_key
        self._market = market
        # Bulk consensus index: {SYMBOL_UPPER: record_dict}
        # Injected from fmp_bulk_prefetch cache to avoid per-ticker API calls.
        self._bulk_consensus_idx: dict = bulk_consensus_idx or {}
        # Ambiguous base symbols (multiple exchange variants) — guard fallback lookups
        self._ambiguous_bases: set = ambiguous_bases or set()

    @property
    def market(self) -> MarketEnum:
        return self._market

    def source_reliability(self, ticker: str) -> float:
        """Returns 1.0 for all markets.

        Regional dampening was removed in v2.2-global. Score compression is
        replaced by the available-factor dynamic denominator in StrategyEngine.
        """
        return 1.0

    def prepare(self, tickers: list[str]) -> list[TickerEntry]:
        """Fetch ALL available factors for EU/Asia tickers via FMP Ultimate.

        Returns TickerEntry list with raw_factors containing every factor
        that has a global FMP data source. congress_score is always 0.0.
        """
        from src.services.fmp_client import (  # noqa: PLC0415
            FMPClient, fmp_prices_to_arrays,
        )
        from src.scoring.momentum_signals import (  # noqa: PLC0415
            score_momentum_long, score_volume_attention,
            score_quality_piotroski, score_price_target_upside,
        )
        from src.scoring.news_signals import (  # noqa: PLC0415
            score_news_sentiment, score_news_buzz,
        )
        from src.scoring.insider_signals import (  # noqa: PLC0415
            score_insider_conviction, score_insider_breadth,
            orthogonalize_insider_scores,
        )
        from src.scoring.analyst import _score_record as ac_score_record  # noqa: PLC0415

        client = FMPClient(api_key=self._api_key)
        entries: list[TickerEntry] = []

        for ticker in tickers:
            try:
                rf = self._fetch_all_factors(
                    ticker, client,
                    score_momentum_long, score_volume_attention,
                    score_quality_piotroski, score_price_target_upside,
                    score_news_sentiment, score_news_buzz,
                    score_insider_conviction, score_insider_breadth,
                    orthogonalize_insider_scores,
                    ac_score_record,
                    fmp_prices_to_arrays,
                )
                if rf is None:
                    continue

                # ── v3.0 shadow columns (SCORING_V3_SHADOW=1) ────────────
                # v2.2 factors untouched; failures degrade to unavailable.
                if _v3_shadow_enabled():
                    try:
                        rf.update(self._v3_intl_columns(ticker, client))
                    except Exception as v3_exc:
                        logger.warning(
                            "FMPFetcher v3 shadow %s: %s — v3 factors "
                            "unavailable for this ticker", ticker, v3_exc)

                entries.append(TickerEntry(
                    ticker=ticker,
                    market=self._market,
                    # sector/cap_tier populated only by the v3 shadow path
                    # (required for engine_v3 bucketing; "" keeps v2.2 stable)
                    sector=rf.get("_v3_sector", "") or "",
                    cap_tier=rf.get("_v3_cap_tier", "") or "",
                    source_reliability=self.source_reliability(ticker),
                    raw_factors=rf,
                ))

            except Exception as exc:
                logger.warning("FMPFetcher: skip %s: %s", ticker, exc)

        return entries

    def _v3_intl_columns(self, ticker: str, client: Any) -> dict:
        """v3.0 EU/ASIA factor columns (plan step 5; cached client calls).

        New factors: inst_concentration, dividend_sustain, margin_expansion
        (discrete-quarter validation + annual fallback), revision_velocity.
        Plus 0.5-centered analyst_revision_score_v3 and an uncoerced
        price_target_upside_score_v3 (the v2.2 key stores 0.0 for missing —
        a downward bias on a signed factor). Sector/cap_tier from the cached
        quote feed engine_v3 bucketing via _v3_sector/_v3_cap_tier.
        """
        from src.ingestion.v3_shadow import _g  # noqa: PLC0415
        from src.scoring.alt_signals import (  # noqa: PLC0415
            score_dividend_sustain, score_inst_concentration,
        )
        from src.scoring.consensus_signals import (  # noqa: PLC0415
            score_analyst_revision, score_revision_velocity,
        )
        from src.scoring.fundamental_signals import score_margin_expansion  # noqa: PLC0415

        cols: dict = {}

        ratios = client.get_ratios_ttm(ticker) or {}
        cf = client.get_cash_flow_statements(ticker, limit=4) or []
        fcf_ttm = sum(float(r.get("freeCashFlow") or 0.0) for r in cf) if cf else None
        paid_ttm = sum(float(r.get("dividendsPaid") or 0.0) for r in cf) if cf else None
        cols["dividend_sustain_score"] = score_dividend_sustain(
            dividend_yield=_g(ratios, "dividendYieldTTM", "dividendYield"),
            payout_ratio=_g(ratios, "dividendPayoutRatioTTM", "payoutRatioTTM",
                            "payoutRatio"),
            fcf_ttm=fcf_ttm,
            dividends_paid_ttm=paid_ttm,
        )

        cols["inst_concentration_score"] = score_inst_concentration(
            client.get_institutional_ownership(ticker))

        quarters: list = []
        for sym in _symbol_candidates(ticker):
            quarters = client.get_income_statements(
                sym, period="quarter", limit=8) or []
            if quarters:
                break
        margin = score_margin_expansion(quarters, [])
        if margin is None:
            annual: list = []
            for sym in _symbol_candidates(ticker):
                annual = client.get_income_statements(
                    sym, period="annual", limit=2) or []
                if annual:
                    break
            margin = score_margin_expansion(quarters, annual)
        cols["margin_expansion_score"] = margin

        cols["revision_velocity_score"] = score_revision_velocity(
            client.get_analyst_estimates(ticker, period="quarter", limit=6) or [])

        rev_pct, n_analysts = client.get_analyst_estimate_revision(ticker)
        cols["analyst_revision_score_v3"] = score_analyst_revision(
            rev_pct, n_analysts)

        cols["price_target_upside_score_v3"] = None
        for sym in _symbol_candidates(ticker):
            upside = client.get_upside_to_target(sym)
            if upside is not None:
                cols["price_target_upside_score_v3"] = upside
                break

        quote = client.get_quote(ticker) or {}
        cols["_v3_sector"] = (quote.get("sector") or "").strip() or "Unknown"
        mcap = float(quote.get("marketCap") or 0.0)
        cols["_v3_cap_tier"] = (
            "large" if mcap >= 10e9 else "mid" if mcap >= 2e9 else "small"
        )
        return cols

    def _fetch_all_factors(
        self, ticker: str, client: Any,
        score_momentum_long, score_volume_attention,
        score_quality_piotroski, score_price_target_upside,
        score_news_sentiment, score_news_buzz,
        score_insider_conviction, score_insider_breadth,
        orthogonalize_insider_scores,
        ac_score_record,
        fmp_prices_to_arrays,
    ) -> Optional[dict]:
        """Fetch and score all global FMP factors for a single ticker.

        Returns None if price data is unavailable (insufficient history / not listed).
        All other factor failures return 0.0 (dead signal) — not None.
        """
        # ── 1. Prices — momentum + volume ────────────────────────────────────
        rows = client.get_historical_prices(ticker, limit=_PRICE_LIMIT)
        if not rows and "." in ticker:
            rows = client.get_historical_prices(
                ticker.split(".")[0], limit=_PRICE_LIMIT)
        closes, volumes, _ = fmp_prices_to_arrays(rows)

        if len(closes) < 5:
            logger.warning(
                "FMPFetcher: no price data for %s — skipping", ticker)
            return None

        return_12_1m: Optional[float] = None
        if len(closes) >= _MIN_BARS_MOMENTUM:
            idx_far = max(0, len(closes) - _MIN_BARS_MOMENTUM)
            idx_near = max(1, len(closes) - 21)
            p_far, p_near = closes[idx_far], closes[idx_near]
            return_12_1m = (p_near - p_far) / p_far if p_far != 0 else None

        volume_spike = 0.0
        n_vol = len(volumes)
        if n_vol > _VOL_BASELINE_SKIP + 5:
            baseline_end = max(0, n_vol - _VOL_BASELINE_SKIP)
            baseline_start = max(0, baseline_end - _VOL_BASELINE_BARS)
            avg_vol = sum(volumes[baseline_start:baseline_end]
                          ) / max(1, baseline_end - baseline_start)
            last_vol = volumes[-1]
            if avg_vol > 0:
                volume_spike = min(last_vol / avg_vol, _VOL_MAX_SPIKE)

        momentum_long_score = score_momentum_long(
            return_12_1m, spy_return_12_1m=0.0)
        volume_attention_score = score_volume_attention(volume_spike)

        # ── 1b. FMP coverage guard ────────────────────────────────────────────
        # Fetch quote early. If FMP has no quote for this ticker it has no
        # coverage — return None for all FMP-sourced factors so StrategyEngine
        # excludes them from the score denominator entirely.
        quote = client.get_quote(ticker)
        if not quote and "." in ticker:
            quote = client.get_quote(ticker.split(".")[0])
        if not quote:
            logger.warning(
                "FMPFetcher: no quote for %s — all FMP factors absent from denominator", ticker)
            return _build_price_only_factors(
                closes, volumes, return_12_1m, volume_spike,
                momentum_long_score, volume_attention_score,
            )
        mktcap_from_quote = float(quote.get("marketCap", 0) or 0)
        sector_from_quote = (quote.get("sector") or "").strip()

        # ── 2. News — sentiment + buzz ────────────────────────────────────────
        news_sentiment_score: Optional[float] = None
        news_buzz_score: Optional[float] = None
        for _sym in _symbol_candidates(ticker):
            try:
                articles = client.get_news_raw_articles(_sym)
                if articles is not None:
                    news_sentiment_score = score_news_sentiment(articles) if articles else 0.0
                    news_buzz_score = score_news_buzz(articles) if articles else 0.0
                    break
            except Exception as exc:
                logger.warning(
                    "FMPFetcher news ABSENT %s (%s): %s(%s) — will try next candidate",
                    ticker, _sym, type(exc).__name__, str(exc)[:80],
                )
        if news_sentiment_score is None:
            logger.warning(
                "FMPFetcher news ABSENT %s: all candidates failed — excluded from denominator", ticker)

        # ── 3. Insider — conviction + breadth ────────────────────────────────
        # FMP stable/insider-trading/search covers:
        #   US:   SEC Form 4 (STOCK Act) — full coverage
        #   EU:   MAR Art.19 mandatory disclosures — good large-cap coverage
        #   Asia: EDINET (JP partial), KRX (KR partial), HKEX (HK partial) — sparse
        #
        # The previous `"." not in ticker` gate was incorrect. market_config.py
        # explicitly lists insider_conviction_score as available for EUROPE and ASIA.
        # It has been removed.
        #
        # Return semantics:
        #   None → excluded from StrategyEngine denominator
        #          (API failure, or suffixed ticker with zero transactions —
        #           cannot distinguish "no coverage" from "no buying")
        #   0.0  → included in denominator as genuine zero signal
        #          (confirmed endpoint response, legitimately no purchases)
        insider_conviction_score: Optional[float] = None
        insider_breadth_score: Optional[float] = None
        is_suffixed = "." in ticker
        total_usd: float = 0.0

        try:
            mktcap = mktcap_from_quote
            total_usd, days_since = client.get_insider_purchases(
                ticker, lookback_days=180)
            btx = client.get_insider_transactions(ticker, lookback_days=90)

            # Endpoint responded → 0.0 (genuine zero, not absent)
            insider_conviction_score = 0.0
            insider_breadth_score = 0.0

            if total_usd > 0 and mktcap > 0:
                insider_conviction_score = score_insider_conviction(
                    key_purchases_usd=total_usd,
                    market_cap=mktcap,
                    days_since_most_recent=days_since,
                )
                logger.info(
                    "FMPFetcher: insider $%.0f for %s (%s)",
                    total_usd, ticker,
                    "MAR Art.19 / local disclosure" if is_suffixed else "SEC Form 4",
                )

            breadth_raw = score_insider_breadth(
                btx.get("P", []), btx.get("S", []))
            insider_conviction_score, insider_breadth_score = \
                orthogonalize_insider_scores(insider_conviction_score, breadth_raw)

            # Sparse coverage guard for EU/Asia:
            # Zero transactions for a suffixed ticker → downgrade to None.
            # StrategyEngine will redistribute insider weight to live factors
            # (news, momentum, analyst) rather than scoring a forced 0.0.
            if is_suffixed \
                    and total_usd == 0 \
                    and not btx.get("P") \
                    and not btx.get("S"):
                insider_conviction_score = None
                insider_breadth_score = None
                logger.debug(
                    "FMPFetcher: zero insider transactions for %s — "
                    "returning None (sparse EU/Asia coverage, not confirmed zero signal)",
                    ticker,
                )

        except Exception as exc:
            logger.warning(
                "FMPFetcher insider ABSENT %s: %s(%s) — excluded from denominator",
                ticker, type(exc).__name__, str(exc)[:80],
            )
            insider_conviction_score = None
            insider_breadth_score = None

        # ── 4. Analyst consensus — bulk index with symbol-candidate fallback ───
        analyst_consensus_score: Optional[float] = None
        try:
            _base_sym = ticker.split(".")[0].upper()
            bulk_rec = self._bulk_consensus_idx.get(ticker.upper())
            if not bulk_rec and _base_sym not in self._ambiguous_bases:
                bulk_rec = self._bulk_consensus_idx.get(_base_sym)
            elif not bulk_rec and _base_sym in self._ambiguous_bases:
                logger.debug(
                    "Skipping ambiguous base alias %s — multiple exchange variants present",
                    _base_sym)
            analyst_consensus_score = 0.0
            if bulk_rec:
                analyst_consensus_score, _ = ac_score_record(ticker, bulk_rec)
            else:
                for _sym in _symbol_candidates(ticker):
                    ratings = client.get_analyst_ratings(_sym)
                    if ratings:
                        analyst_consensus_score, _ = ac_score_record(ticker, ratings)
                        break
        except Exception as exc:
            logger.warning(
                "FMPFetcher analyst_consensus ABSENT %s: %s(%s) — excluded from denominator",
                ticker, type(exc).__name__, str(exc)[:80],
            )

        # ── 5. Analyst revision momentum ──────────────────────────────────────
        analyst_revision_score = None  # None = API failure; 0.0 = no revision signal
        analyst_revision_n = 0         # coverage count for Discord catalyst display
        try:
            rev_pct, rev_n = client.get_analyst_estimate_revision(ticker)
            analyst_revision_score = 0.0
            analyst_revision_n = int(rev_n or 0)
            if rev_pct is not None and rev_n >= 3:
                clipped = max(-0.30, min(0.30, rev_pct))
                analyst_revision_score = round(
                    ((clipped + 0.30) / 0.60) * min(1.0, rev_n / 10.0), 4
                )
        except Exception as exc:
            logger.warning(
                "FMPFetcher analyst_revision ABSENT %s: %s(%s) — analyst_revision excluded from denominator",
                ticker, type(exc).__name__, str(exc)[:80],
            )

        # ── 6. Quality — Piotroski F-Score ────────────────────────────────────
        quality_piotroski_score = None  # None = API failure; 0.0 = no quality signal
        try:
            ratios = client.get_ratios_ttm(ticker)
            if not ratios and "." in ticker:
                # ratios-ttm-bulk indexes by base symbol (e.g. ASML not ASML.AS)
                ratios = client.get_ratios_ttm(ticker.split(".")[0])
            quality_piotroski_score = 0.0  # API call succeeded
            if ratios:
                quality_piotroski_score, _quality_piotroski_raw = score_quality_piotroski(ratios)
        except Exception as exc:
            logger.warning(
                "FMPFetcher piotroski ABSENT %s: %s(%s) — quality_piotroski excluded from denominator",
                ticker, type(exc).__name__, str(exc)[:80],
            )

        # ── 7. Price target upside — symbol-candidate fallback ────────────────
        price_target_upside_score: Optional[float] = None
        raw_target_price = None
        raw_current_price = quote.get("price")
        try:
            pt_data = None
            for _sym in _symbol_candidates(ticker):
                pt_data = client.get_price_target_consensus(_sym)
                if pt_data:
                    break
            price_target_upside_score = 0.0
            raw_target_price = pt_data.get("targetConsensus") if pt_data else None
            for _sym in _symbol_candidates(ticker):
                upside = client.get_upside_to_target(_sym)
                if upside is not None:
                    price_target_upside_score = upside
                    break
        except Exception as exc:
            logger.warning(
                "FMPFetcher price_target ABSENT %s: %s(%s) — excluded from denominator",
                ticker, type(exc).__name__, str(exc)[:80],
            )

        # ── 8. Market cap ─────────────────────────────────────────────────────
        mktcap_final = mktcap_from_quote

        # ── 8b. Earnings surprise (for Discord PEAD badge on EU/Asia) ────────
        _eps_pct_intl: Optional[float] = None
        _eps_days_intl: int = 0
        try:
            _eps_pct_intl, _eps_days_intl = client.get_earnings_surprise(ticker)
        except Exception:
            pass  # non-critical — PEAD badge just won't show

        # ── 9. FCF Yield — Damodaran value signal ────────────────────────────
        from src.scoring.fundamental_signals import (  # noqa: PLC0415
            score_fcf_yield, score_amihud_shock, score_pb_value_up, score_roic_quality,
        )
        fcf_yield_score: Optional[float] = None
        try:
            ev = client.get_enterprise_value(ticker)
            if ev and ev > 0:
                cf_stmts = client.get_cash_flow_statements(ticker) or []
                ttm_fcf = sum(float(q.get("freeCashFlow", 0) or 0) for q in cf_stmts[:4])
                if ttm_fcf > 0:
                    fcf_yield_score = score_fcf_yield(ttm_fcf, ev)
                else:
                    fcf_yield_score = 0.0
            else:
                fcf_yield_score = 0.0
        except Exception as exc:
            logger.warning(
                "FMPFetcher fcf_yield ABSENT %s: %s(%s) — excluded from denominator",
                ticker, type(exc).__name__, str(exc)[:80],
            )

        # ── 10. Amihud Illiquidity Shock — zero new API calls ─────────────────
        amihud_shock_score: float = 0.0
        try:
            amihud_shock_score = score_amihud_shock(
                price_history=closes[-25:],
                volume_history=volumes[-25:],
            )
        except Exception as exc:
            logger.warning("FMPFetcher amihud_shock ABSENT %s: %s", ticker, exc)

        # ── 11. Dynamic P/B — Fama & French value trigger ─────────────────────
        pb_value_up_score: Optional[float] = None
        try:
            if not ratios and "." in ticker:
                ratios = client.get_ratios_ttm(ticker.split(".")[0])
            bvps  = float((ratios or {}).get("bookValuePerShareTTM") or 0)
            price = float((quote or {}).get("price") or 0)
            if bvps > 0 and price > 0:
                pb_value_up_score = score_pb_value_up(bvps, price)
            else:
                pb_value_up_score = 0.0
        except Exception as exc:
            logger.warning(
                "FMPFetcher pb_value_up ABSENT %s: %s(%s) — excluded from denominator",
                ticker, type(exc).__name__, str(exc)[:80],
            )

        # ── 12. ROIC / ROE Quality — Greenblatt magic formula ─────────────────
        roic_quality_score: Optional[float] = None
        try:
            roe  = float((ratios or {}).get("returnOnEquityTTM") or 0)
            roce_raw = (ratios or {}).get("returnOnCapitalEmployedTTM")
            roce: Optional[float] = float(roce_raw) if roce_raw is not None else None
            if roe != 0:
                roic_quality_score = score_roic_quality(roe, roce)
            else:
                roic_quality_score = 0.0
        except Exception as exc:
            logger.warning(
                "FMPFetcher roic_quality ABSENT %s: %s(%s) — excluded from denominator",
                ticker, type(exc).__name__, str(exc)[:80],
            )

        return {
            "momentum_long_score":       momentum_long_score,
            "volume_attention_score":    volume_attention_score,
            "news_sentiment_score":      news_sentiment_score,
            "news_buzz_score":           news_buzz_score,
            "insider_conviction_score":  insider_conviction_score,
            "insider_breadth_score":     insider_breadth_score,
            "analyst_consensus_score":   analyst_consensus_score,
            "analyst_revision_score":    analyst_revision_score,
            "quality_piotroski_score":   quality_piotroski_score,
            "price_target_upside_score": price_target_upside_score,
            # EU/Asia fundamental value + quality signals
            "fcf_yield_score":           fcf_yield_score,
            "amihud_shock_score":        amihud_shock_score,
            "pb_value_up_score":         pb_value_up_score,
            "roic_quality_score":        roic_quality_score,
            # Structurally absent — always 0.0
            "congress_score":            0.0,
            "transcript_tone_score":     0.0,
            # Raw inputs (diagnostic)
            "return_12_1m":              return_12_1m,
            "volume_spike":              volume_spike,
            "market_cap":                mktcap_final,
            "target_price":              raw_target_price,
            "current_price":             raw_current_price,
            "news_sentiment_source":     "fmp" if (news_sentiment_score or 0) > 0 else "none",
            "news_buzz_source":          "fmp" if (news_buzz_score or 0) > 0 else "none",
            "analyst_consensus_source":  "bulk" if (analyst_consensus_score or 0) > 0 else "none",
            "insider_source":            "fmp" if (insider_conviction_score or 0) > 0 else "none",
            # Raw values for Discord display (not scoring inputs)
            "insider_usd":               float(total_usd) if total_usd else 0.0,
            "earnings_surprise_pct":     _eps_pct_intl,
            "earnings_surprise_days":    _eps_days_intl,
            "analyst_revision_n_analysts": analyst_revision_n,
            # Company meta (sector for Discord sector heatmap)
            "sector":                    sector_from_quote,
        }


def _symbol_candidates(ticker: str) -> list[str]:
    """Ordered list of symbol formats to try against FMP endpoints.

    FMP indexes EU/Asia tickers inconsistently across endpoints:
    - Some accept the exchange-suffixed form (ASML.AS, SAP.DE)
    - Others only respond to the base symbol (ASML, SAP)
    Always try the most specific form first, fall back to base.
    """
    candidates = [ticker]
    if "." in ticker:
        base = ticker.split(".")[0]
        if base != ticker:
            candidates.append(base)
    return candidates


def _build_price_only_factors(
    closes: list,
    volumes: list,
    return_12_1m: Optional[float],
    volume_spike: float,
    momentum_long_score: float,
    volume_attention_score: float,
) -> dict:
    """Factor dict with only price-derived scores; all FMP-sourced factors are None.

    Used when FMP has no quote coverage for a ticker — we still have price history
    from the historical-prices endpoint, so momentum and volume are valid.
    All other factors are None so StrategyEngine excludes them from the denominator.
    """
    return {
        "momentum_long_score":       momentum_long_score,
        "volume_attention_score":    volume_attention_score,
        "news_sentiment_score":      None,
        "news_buzz_score":           None,
        "insider_conviction_score":  None,
        "insider_breadth_score":     None,
        "analyst_consensus_score":   None,
        "analyst_revision_score":    None,
        "quality_piotroski_score":   None,
        "price_target_upside_score": None,
        "fcf_yield_score":           None,
        "amihud_shock_score":        0.0,
        "pb_value_up_score":         None,
        "roic_quality_score":        None,
        "congress_score":            0.0,
        "transcript_tone_score":     0.0,
        "return_12_1m":              return_12_1m,
        "volume_spike":              volume_spike,
        "market_cap":                0.0,
        "target_price":              None,
        "current_price":             None,
        "quality_piotroski_raw":     None,
        "news_sentiment_source":     "none",
        "news_buzz_source":          "none",
        "analyst_consensus_source":  "none",
        "insider_source":            "none",
    }
