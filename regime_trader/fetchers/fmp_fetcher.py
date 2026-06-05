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
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .base import BaseMarketFetcher, MarketEnum, TickerEntry

logger = logging.getLogger(__name__)

_VOL_BASELINE_BARS = 90
_VOL_BASELINE_SKIP = 5
_VOL_MAX_SPIKE     = 20.0
_MIN_BARS_MOMENTUM = 252
_PRICE_LIMIT       = 280   # 13 months of trading days


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
    ) -> None:
        self._api_key = api_key
        self._market  = market
        # Bulk consensus index: {SYMBOL_UPPER: record_dict}
        # Injected from fmp_bulk_prefetch cache to avoid per-ticker API calls.
        self._bulk_consensus_idx: dict = bulk_consensus_idx or {}

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

        client  = FMPClient(api_key=self._api_key)
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
            rows = client.get_historical_prices(ticker.split(".")[0], limit=_PRICE_LIMIT)
        closes, volumes, _ = fmp_prices_to_arrays(rows)

        if len(closes) < 5:
            logger.warning("FMPFetcher: no price data for %s — skipping", ticker)
            return None

        return_12_1m: Optional[float] = None
        if len(closes) >= _MIN_BARS_MOMENTUM:
            idx_far  = max(0, len(closes) - _MIN_BARS_MOMENTUM)
            idx_near = max(1, len(closes) - 21)
            p_far, p_near = closes[idx_far], closes[idx_near]
            return_12_1m = (p_near - p_far) / p_far if p_far != 0 else None

        volume_spike = 0.0
        n_vol = len(volumes)
        if n_vol > _VOL_BASELINE_SKIP + 5:
            baseline_end   = max(0, n_vol - _VOL_BASELINE_SKIP)
            baseline_start = max(0, baseline_end - _VOL_BASELINE_BARS)
            avg_vol  = sum(volumes[baseline_start:baseline_end]) / max(1, baseline_end - baseline_start)
            last_vol = volumes[-1]
            if avg_vol > 0:
                volume_spike = min(last_vol / avg_vol, _VOL_MAX_SPIKE)

        momentum_long_score    = score_momentum_long(return_12_1m, spy_return_12_1m=0.0)
        volume_attention_score = score_volume_attention(volume_spike)

        # ── 2. News — sentiment + buzz ────────────────────────────────────────
        news_sentiment_score = 0.0
        news_buzz_score      = 0.0
        try:
            articles = client.get_news_raw_articles(ticker)
            if not articles and "." in ticker:
                # FMP news/stock indexes by base symbol (e.g. ASML not ASML.AS)
                articles = client.get_news_raw_articles(ticker.split(".")[0])
            if articles:
                s = score_news_sentiment(articles)
                if s > 0.0:
                    news_sentiment_score = s
                news_buzz_score = score_news_buzz(articles)
        except Exception as exc:
            logger.debug("FMPFetcher news %s: %s", ticker, exc)

        # ── 3. Insider — conviction + breadth ────────────────────────────────
        insider_conviction_score = 0.0
        insider_breadth_score    = 0.0
        try:
            quote  = client.get_quote(ticker)
            mktcap = float(quote.get("marketCap", 0) or 0)
            total_usd, days_since = client.get_insider_purchases(ticker, lookback_days=180)
            btx = client.get_insider_transactions(ticker, lookback_days=90)
            if total_usd > 0 and mktcap > 0:
                insider_conviction_score = score_insider_conviction(
                    key_purchases_usd=total_usd,
                    market_cap=mktcap,
                    days_since_most_recent=days_since,
                )
            breadth_raw = score_insider_breadth(btx.get("P", []), btx.get("S", []))
            insider_conviction_score, insider_breadth_score = orthogonalize_insider_scores(
                insider_conviction_score, breadth_raw
            )
        except Exception as exc:
            logger.debug("FMPFetcher insider %s: %s", ticker, exc)

        # ── 4. Analyst consensus — bulk index lookup ──────────────────────────
        analyst_consensus_score = 0.0
        try:
            _base_sym = ticker.split(".")[0].upper()
            bulk_rec = (
                self._bulk_consensus_idx.get(ticker.upper())
                or self._bulk_consensus_idx.get(_base_sym)
            )
            if bulk_rec:
                analyst_consensus_score, _ = ac_score_record(ticker, bulk_rec)
            else:
                ratings = client.get_analyst_ratings(ticker)
                if ratings:
                    analyst_consensus_score, _ = ac_score_record(ticker, ratings)
        except Exception as exc:
            logger.debug("FMPFetcher analyst_consensus %s: %s", ticker, exc)

        # ── 5. Analyst revision momentum ──────────────────────────────────────
        analyst_revision_score = 0.0
        try:
            from scripts.run_pipeline import score_analyst_revision  # noqa: PLC0415
            rev_pct, rev_n = client.get_analyst_estimate_revision(ticker)
            analyst_revision_score = score_analyst_revision(rev_pct, rev_n)
        except Exception as exc:
            logger.debug("FMPFetcher analyst_revision %s: %s", ticker, exc)

        # ── 6. Quality — Piotroski F-Score ────────────────────────────────────
        quality_piotroski_score = 0.0
        try:
            ratios = client.get_ratios_ttm(ticker)
            if not ratios and "." in ticker:
                # ratios-ttm-bulk indexes by base symbol (e.g. ASML not ASML.AS)
                ratios = client.get_ratios_ttm(ticker.split(".")[0])
            if ratios:
                quality_piotroski_score = score_quality_piotroski(ratios)
        except Exception as exc:
            logger.debug("FMPFetcher piotroski %s: %s", ticker, exc)

        # ── 7. Price target upside ────────────────────────────────────────────
        price_target_upside_score = 0.0
        raw_target_price  = None
        raw_current_price = None
        try:
            pt_data = client.get_price_target_consensus(ticker)
            if not pt_data and "." in ticker:
                pt_data = client.get_price_target_consensus(ticker.split(".")[0])
            if not locals().get("quote"):
                quote = client.get_quote(ticker)
            raw_target_price  = pt_data.get("targetConsensus") if pt_data else None
            raw_current_price = quote.get("price") if quote else None
            upside = client.get_upside_to_target(ticker)
            if upside is None and "." in ticker:
                upside = client.get_upside_to_target(ticker.split(".")[0])
            if upside is not None:
                price_target_upside_score = upside
        except Exception as exc:
            logger.debug("FMPFetcher price_target %s: %s", ticker, exc)

        # ── 8. Market cap ─────────────────────────────────────────────────────
        mktcap_final = 0.0
        try:
            if not locals().get("quote"):
                quote = client.get_quote(ticker)
            mktcap_final = float(quote.get("marketCap", 0) or 0)
        except Exception:
            pass

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
            "news_sentiment_source":     "fmp" if news_sentiment_score > 0 else "none",
            "news_buzz_source":          "fmp" if news_buzz_score > 0 else "none",
            "analyst_consensus_source":  "bulk" if analyst_consensus_score > 0 else "none",
            "insider_source":            "fmp" if insider_conviction_score > 0 else "none",
        }
