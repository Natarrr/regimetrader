# Path: regime_trader/fetchers/fmp_fetcher.py
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
        from regime_trader.services.fmp_client import (  # noqa: PLC0415
            FMPClient, fmp_prices_to_arrays,
        )
        from regime_trader.scoring.momentum_signals import (  # noqa: PLC0415
            score_momentum_long, score_volume_attention,
            score_quality_piotroski, score_price_target_upside,
        )
        from regime_trader.scoring.news_signals import (  # noqa: PLC0415
            score_news_sentiment, score_news_buzz,
        )
        from regime_trader.scoring.insider_signals import (  # noqa: PLC0415
            score_insider_conviction, score_insider_breadth,
            orthogonalize_insider_scores,
        )
        from regime_trader.scoring.analyst import _score_record as ac_score_record  # noqa: PLC0415

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

                entries.append(TickerEntry(
                    ticker=ticker,
                    market=self._market,
                    sector="",
                    cap_tier="",
                    source_reliability=self.source_reliability(ticker),
                    raw_factors=rf,
                ))

            except Exception as exc:
                logger.warning("FMPFetcher: skip %s: %s", ticker, exc)

        return entries

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
        # FMP insider-trading/search is SEC Form 4 (US STOCK Act) ONLY.
        # EU/Asia companies file under MAR Art.19 or local regimes — not in FMP.
        # For non-US tickers, keep None so StrategyEngine excludes these from
        # the denominator (not penalised as genuine zero-insider-buying signal).
        is_us_ticker = "." not in ticker
        insider_conviction_score: Optional[float] = None
        insider_breadth_score: Optional[float] = None

        if is_us_ticker:
            try:
                mktcap = mktcap_from_quote
                total_usd, days_since = client.get_insider_purchases(
                    ticker, lookback_days=180)
                btx = client.get_insider_transactions(ticker, lookback_days=90)
                insider_conviction_score = 0.0
                insider_breadth_score = 0.0
                if total_usd > 0 and mktcap > 0:
                    insider_conviction_score = score_insider_conviction(
                        key_purchases_usd=total_usd,
                        market_cap=mktcap,
                        days_since_most_recent=days_since,
                    )
                breadth_raw = score_insider_breadth(
                    btx.get("P", []), btx.get("S", []))
                insider_conviction_score, insider_breadth_score = orthogonalize_insider_scores(
                    insider_conviction_score, breadth_raw
                )
            except Exception as exc:
                logger.warning(
                    "FMPFetcher insider ABSENT %s: %s(%s) — excluded from denominator",
                    ticker, type(exc).__name__, str(exc)[:80],
                )
                insider_conviction_score = None
                insider_breadth_score = None
        else:
            logger.debug(
                "FMPFetcher insider SKIP %s — non-US ticker, SEC Form 4 not applicable", ticker)

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
        try:
            rev_pct, rev_n = client.get_analyst_estimate_revision(ticker)
            analyst_revision_score = 0.0
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
