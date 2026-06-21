# Path: src/services/fmp/estimates.py
"""EstimatesSentimentMixin — estimates endpoint methods for FMPClient."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from src.services.fmp.core import FMPEndpointError

log = logging.getLogger(__name__)


class EstimatesSentimentMixin:

    def get_news_raw_articles(self, ticker: str) -> List[Dict]:
        """Return raw news articles for sentiment+buzz scoring (cached 2h).

        Falls back to base symbol (strips exchange suffix) when dotted ticker
        returns no results — FMP news API may not index EU/Asia suffixed tickers.
        """
        if not self._api_key:
            return []
        cached = self._cache_read("news", ticker)
        if cached is not None:
            return cached
        data = self._get("news/stock", {"symbols": ticker, "limit": 50},
                         bucket="news") or []
        result: List[Dict] = data if isinstance(data, list) else []
        if not result and "." in ticker:
            base = ticker.split(".")[0]
            cached_base = self._cache_read("news", base)
            if cached_base is not None:
                return cached_base
            data2 = self._get("news/stock", {"symbols": base, "limit": 50},
                              bucket="news") or []
            result = data2 if isinstance(data2, list) else []
            if result:
                self._cache_write("news", base, result)
        if result:
            self._cache_write("news", ticker, result)
        return result

    def get_earnings_surprise(self, ticker: str) -> Tuple[Optional[float], int]:
        """Return (surprise_pct, days_since) for the most recent quarter.

        Post-Earnings Announcement Drift (PEAD): Bernard & Thomas (1989, JAE)
        showed that standardized unexpected earnings (SUE) predict returns for
        60–90 days post-announcement — the most robust anomaly in event studies.

        surprise_pct = (epsActual - epsEstimated) / abs(epsEstimated)
        days_since   = calendar days from the announcement date to today

        Source: stable/ "earnings" (the legacy "earnings-surprises" route is
        HTTP 404 on stable/). The earnings calendar mixes future scheduled
        quarters (epsActual=None) with past reports, newest-first; the most
        recent PAST report with both actual and estimated EPS is used.

        Returns (None, 0) gracefully on any error, empty response, or zero estimate
        (avoids division-by-zero on pre-revenue companies).

        Uses the "news" TTL bucket — earnings surprise data changes at most
        once per quarter so the news TTL is conservative and keeps the cache coherent.
        """
        if not self._api_key:
            return None, 0

        cache_key = f"eps_surprise_{ticker}"
        cached = self._cache_read("news", cache_key)
        if cached is not None:
            return tuple(cached)  # type: ignore[return-value]

        try:
            # limit=8: covers up to ~4 future scheduled quarters plus >=4 past reports.
            data = self._get(
                "earnings",
                {"symbol": ticker, "limit": 8},
                bucket="news",
            ) or []
            if not isinstance(data, list) or not data:
                self._cache_write("news", cache_key, [None, 0])
                return None, 0

            from datetime import date as _date
            today = datetime.now(timezone.utc).date()

            for row in data:
                date_str = str(row.get("date") or "")[:10]
                try:
                    announced = _date.fromisoformat(date_str)
                except ValueError:
                    continue
                if announced > today:
                    continue  # scheduled future quarter — no actuals yet
                actual = row.get("epsActual")
                estimate = row.get("epsEstimated")
                if actual is None or estimate is None:
                    continue
                actual = float(actual)
                estimate = float(estimate)

                # Guard: zero or near-zero estimate → undefined surprise % (pre-revenue)
                if abs(estimate) < 1e-6:
                    break

                surprise_pct = (actual - estimate) / abs(estimate)
                days_since = max(0, (today - announced).days)
                result = (round(surprise_pct, 6), days_since)
                self._cache_write("news", cache_key, list(result))
                return result

            self._cache_write("news", cache_key, [None, 0])
            return None, 0

        except FMPEndpointError:
            # Structural failure already logged by _get(); propagate to health_report
            return None, 0
        except Exception as exc:
            log.debug("get_earnings_surprise %s failed: %s", ticker, exc)
            return None, 0

    def get_analyst_estimate_revision(self, ticker: str) -> Tuple[Optional[float], int]:
        """Return (revision_pct, n_analysts) measuring EPS estimate revision momentum.

        Analyst estimate revision momentum is a core quant factor used by AQR,
        Two Sigma, and most systematic equity funds. The intuition is that analysts
        revising EPS estimates upward signal an improving fundamental view that is
        not yet fully reflected in price — orthogonal to price momentum (Jegadeesh-
        Titman 1993) which captures past returns. Academic grounding:

          Chan, Jegadeesh & Lakonishok (1996, JF): "Momentum Strategies" —
          estimate revisions predict future abnormal returns independently of
          past price performance.

        revision_pct = (estimates[0].estimatedEpsAvg - estimates[2].estimatedEpsAvg)
                       / abs(estimates[2].estimatedEpsAvg)

        estimates[0] = most recent quarter, estimates[2] = ~3 quarters ago.
        FMP returns newest-first so index 0 is the freshest estimate.

        n_analysts is taken from estimates[0].numberAnalystEstimatedEps and used
        by the scorer as a coverage weight (thin coverage → low confidence).

        Returns (None, 0) when:
          - No API key
          - Fewer than 3 estimates available (can't compute a revision)
          - Base estimate is zero or near-zero (division guard)
          - Any network / parse error

        Cache bucket: "ratings" (6h TTL) — analyst estimates change slowly, at
        most once per quarter, so 6h is conservative relative to the signal horizon.
        """
        if not self._api_key:
            return None, 0

        cache_key = f"eps_revision_{ticker}"
        cached = self._cache_read("ratings", cache_key)
        if cached is not None:
            return tuple(cached)  # type: ignore[return-value]

        _null = [None, 0]
        symbols_to_try = [ticker]
        if "." in ticker:
            symbols_to_try.append(ticker.split(".")[0])

        for sym in symbols_to_try:
            try:
                data = self._get(
                    "analyst-estimates",
                    {"symbol": sym, "period": "quarter", "limit": 4},
                    bucket="ratings",
                ) or []
                if not isinstance(data, list) or len(data) < 3:
                    continue  # try next symbol

                recent = data[0]
                base_est = data[2]

                recent_eps = recent.get("estimatedEpsAvg")
                base_eps = base_est.get("estimatedEpsAvg")

                if recent_eps is None or base_eps is None:
                    continue

                recent_eps = float(recent_eps)
                base_eps = float(base_eps)

                if abs(base_eps) < 1e-6:
                    continue

                revision_pct = (recent_eps - base_eps) / abs(base_eps)
                n_analysts = int(recent.get("numberAnalystEstimatedEps") or 0)
                result = [round(revision_pct, 6), n_analysts]
                self._cache_write("ratings", cache_key, result)
                return tuple(result)  # type: ignore[return-value]

            except FMPEndpointError:
                self._cache_write("ratings", cache_key, _null)
                return None, 0
            except Exception as exc:
                log.debug("get_analyst_estimate_revision %s (%s) failed: %s", ticker, sym, exc)
                continue

        self._cache_write("ratings", cache_key, _null)
        return None, 0

    def get_analyst_ratings(self, ticker: str) -> Dict:
        """Analyst consensus (stable/grades-consensus). PASS in smoke-test."""
        if not self._api_key:
            return {}
        cached = self._cache_read("ratings", ticker)
        if cached is not None:
            return cached
        data = self._get("grades-consensus",
                         {"symbol": ticker}, bucket="ratings") or []
        result = data[0] if isinstance(data, list) and data else (data or {})
        self._cache_write("ratings", ticker, result)
        return result

    def get_revenue_estimates(self, symbol: str, limit: int = 6) -> list:
        """Revenue estimate history (stable/revenue-estimates). FMP Ultimate.

        Returns newest-first list of quarterly revenue estimate rows, each
        containing estimatedRevenueAvg and numberAnalystEstimatedRevenue.
        Used by score_revenue_revision() [Zacks, 2003].
        """
        if not self._api_key:
            return []
        cache_key = f"rev_est_{symbol}_{limit}"
        cached = self._cache_read("ratings", cache_key)
        if cached is not None:
            return cached
        data = self._get("revenue-estimates",
                         {"symbol": symbol, "limit": limit}, bucket="ratings") or []
        result = data if isinstance(data, list) else []
        self._cache_write("ratings", cache_key, result)
        return result

    def get_recent_upgrades_downgrades(self, ticker: str, lookback_days: int = 7) -> Dict:
        """Fetch recent upgrades/downgrades within lookback_days.

        Returns a dict with keys: action, from_grade, to_grade, analyst_firm,
        days_ago, score_delta. Returns {'action': 'none'} on error or no records.
        """
        if not self._api_key:
            return {"action": "none"}
        cache_key = f"upgrades_{ticker}_{lookback_days}d"
        cached = self._cache_read("ratings", cache_key)
        if cached is not None:
            return cached

        try:
            data = self._get("upgrades-downgrades", {"symbol": ticker, "page": 0}, bucket="ratings") or []
        except FMPEndpointError:
            return {"action": "none"}
        except Exception:
            return {"action": "none"}

        if not isinstance(data, list) or not data:
            return {"action": "none"}

        # Grade score map
        from datetime import date as _dt_date
        _GRADE_SCORE = {
            "strongbuy": 1.0, "buy": 0.75, "outperform": 0.70, "overweight": 0.70,
            "hold": 0.50, "neutral": 0.50, "underperform": 0.25, "sell": 0.10,
            "underweight": 0.10, "strongsell": 0.0,
        }

        best_record = None
        best_days = None
        best_action = None

        for rec in data:
            # date field may be 'publishedDate' or 'date'
            raw_date = str(rec.get("publishedDate") or rec.get("date") or "")[:10]
            try:
                d = _dt_date.fromisoformat(raw_date)
            except Exception:
                continue
            days_ago = (datetime.now(timezone.utc).date() - d).days
            if days_ago > lookback_days:
                continue
            action_raw = str(rec.get("action") or "").lower()
            if "upgrade" in action_raw:
                action = "upgrade"
            elif "downgrade" in action_raw:
                action = "downgrade"
            elif "initiat" in action_raw or "cover" in action_raw:
                action = "initiate"
            else:
                continue

            if best_record is None or days_ago < best_days:
                best_record = rec
                best_days = days_ago
                best_action = action

        if not best_record:
            return {"action": "none"}

        from_grade = best_record.get("fromGrade") or best_record.get("from") or None
        to_grade = best_record.get("toGrade") or best_record.get("to") or None
        firm = best_record.get("analystFirm") or best_record.get("firm") or None

        from_score = _GRADE_SCORE.get(str(from_grade).lower(), None) if from_grade else None
        to_score = _GRADE_SCORE.get(str(to_grade).lower(), None) if to_grade else None
        score_delta = None
        if to_score is not None and from_score is not None:
            score_delta = to_score - from_score

        result = {
            "action": best_action or "none",
            "from_grade": from_grade,
            "to_grade": to_grade,
            "analyst_firm": firm,
            "days_ago": int(best_days) if best_days is not None else None,
            "score_delta": float(score_delta) if score_delta is not None else None,
        }

        try:
            self._cache_write("ratings", cache_key, result)
        except Exception:
            pass
        return result

    def get_analyst_estimates(
        self, ticker: str, period: str = "quarter", limit: int = 6
    ) -> List[Dict]:
        """Raw analyst-estimates rows, newest-first.

        v3.0 revision_velocity input (needs 4 consecutive estimate rows —
        the second derivative of the revision path; get_analyst_estimate_revision
        only exposes the first derivative as a scalar).
        """
        if not self._api_key:
            return []
        data = self._get("analyst-estimates",
                         {"symbol": ticker, "period": period, "limit": limit},
                         bucket="ratings") or []
        return data if isinstance(data, list) else []

    def get_price_target_consensus(self, ticker: str) -> Dict:
        """Price target consensus (stable/price-target-consensus).

        Falls back to base symbol for EU/Asia tickers (e.g. ASML.AS → ASML).
        The returned dict carries "_resolved_symbol" — the symbol variant the
        consensus actually came from — so callers can pair the quote with the
        SAME variant (a USD ADR target must never meet a local-currency quote).
        """
        if not self._api_key:
            return {}
        data = self._get("price-target-consensus", {"symbol": ticker},
                         bucket="ratings") or []
        result = data[0] if isinstance(data, list) and data else {}
        resolved = ticker
        if not result and "." in ticker:
            base = ticker.split(".")[0]
            data2 = self._get("price-target-consensus", {"symbol": base},
                              bucket="ratings") or []
            result = data2[0] if isinstance(data2, list) and data2 else {}
            resolved = base
        if result:
            result = dict(result)
            result["_resolved_symbol"] = resolved
        return result

    def get_upside_to_target(self, ticker: str, max_age_days: int = 90) -> Optional[float]:
        """Analyst consensus price target upside score in [0, 1], or None.

        Computes score_price_target_upside(targetConsensus, currentPrice) over the
        currency-paired (target, price) from _paired_target_and_price() — the same
        pairing/rescue the Discord 🎯 exit-anchor renders, so score and displayed
        level can never disagree.

        Returns None when:
          - No API key
          - targetConsensus or price is missing, zero, or non-numeric
          - Target is older than max_age_days (stale — treated as dead signal)
          - Suspected currency/scale mismatch (post-rescue ratio outside [0.2, 5.0])
          - Either delegated call raises an exception

        None → caller converts to 0.0 via `or 0.0` → dead signal penalized
        in cross-sectional normalization. Distinct from 0.50 (at-target, valid data).
        """
        from src.scoring.momentum_signals import score_price_target_upside  # noqa: PLC0415
        paired = self._paired_target_and_price(ticker, max_age_days=max_age_days)
        if paired is None:
            return None
        target_f, price_f, _ = paired
        return score_price_target_upside(target_f, price_f)

    def _paired_target_and_price(
        self, ticker: str, max_age_days: int = 90
    ) -> Optional[Tuple[float, float, str]]:
        """Currency-paired analyst target + spot price, or None.

        SINGLE SOURCE OF TRUTH behind both get_upside_to_target (the score) and
        the Discord 🎯 exit-anchor (the displayed level) — so the two can never
        disagree (the SHEL.L "$102 (−96.6%)" class of bug).

        Pairs the consensus target with the quote from the SAME resolved symbol
        variant (a USD ADR target must never meet a local-currency quote),
        applies the LSE GBX/GBP 100× rescue, enforces the staleness cutoff
        (max_age_days), and a [0.2, 5.0] order-of-magnitude backstop (residual
        currency mixing → None).

        Returns (target, price, resolved_symbol) — target and price in the SAME
        currency unit. None on: no API key, missing/zero/non-numeric target or
        price, stale target, suspected currency mismatch, or any exception.
        """
        if not self._api_key:
            return None
        try:
            from datetime import date as _date  # noqa: PLC0415
            target_data = self.get_price_target_consensus(ticker)
            # Same-symbol pairing: the quote MUST come from the same symbol
            # variant the PT consensus resolved to (ASML.AS → ASML fallback
            # means USD target × USD ADR quote, never USD target × EUR quote).
            quote_symbol = target_data.get("_resolved_symbol", ticker)
            quote_data = self.get_quote(quote_symbol)
            target = target_data.get("targetConsensus")
            price = quote_data.get("price")
            if not target or not price:
                return None

            target_date_str = (
                target_data.get("targetConsensusDate")
                or target_data.get("lastUpdated")
                or ""
            )
            if target_date_str:
                try:
                    target_date = _date.fromisoformat(str(target_date_str)[:10])
                    age_days = (datetime.now(timezone.utc).date() - target_date).days
                    if age_days > max_age_days:
                        log.debug(
                            "_paired_target_and_price %s: target is %dd old (> %dd threshold) — "
                            "returning None (stale, treated as dead signal)",
                            ticker, age_days, max_age_days,
                        )
                        return None
                except Exception:
                    pass  # unparseable date — proceed without staleness filter

            target_f = float(target)
            price_f = float(price)
            ratio = target_f / price_f
            # GBX/GBP rescue: LSE lines quote in pence while consensus targets
            # are often published in pounds (structural 100× hazard). Rescue
            # recovers the signal instead of None-dropping the UK book. After
            # rescue, target and price are both in pence (GBX).
            if quote_symbol.upper().endswith(".L"):
                if 0.005 <= ratio <= 0.02:
                    target_f *= 100.0
                elif 50.0 <= ratio <= 200.0:
                    target_f /= 100.0
                ratio = target_f / price_f
            # Order-of-magnitude backstop (all symbols): post-rescue scale
            # mismatch means residual currency mixing — None, never a level.
            if ratio > 5.0 or ratio < 0.2:
                log.warning(
                    "_paired_target_and_price %s: target/price ratio %.3f outside "
                    "[0.2, 5.0] — suspected currency/scale mismatch, returning None",
                    ticker, ratio,
                )
                return None
            return (target_f, price_f, quote_symbol)
        except Exception as exc:
            log.debug("_paired_target_and_price %s failed: %s", ticker, exc)
            return None

    def get_earnings_transcript(self, ticker: str, max_chars: int = 3000) -> Optional[str]:
        """Executive remarks from the most recent earnings call.

        Fetches stable/earning-call-transcript-latest (limit=1).
        Returns content[:max_chars] on success; None on any failure.

        max_chars (default 3000) is intentionally larger than build_prompt's
        transcript_max_chars (default 2000) — the delta sits in memory and is
        discarded. This avoids a second network call if the prompt budget changes.

        Cache bucket: "transcript" (24h TTL — transcripts don't change after
        publication). Soft-fail: FMPEndpointError and network exceptions return
        None; the transcript is additive enrichment, not a scored factor.
        """
        if not self._api_key:
            return None
        cached = self._cache_read("transcript", ticker)
        if cached is not None:
            return cached
        try:
            data = self._get(
                "earning-call-transcript-latest",
                {"symbol": ticker, "limit": 1},
                bucket="transcript",
            ) or []
            if not isinstance(data, list) or not data:
                return None
            content = data[0].get("content")
            if not content:
                # FMP returned a record but without transcript text yet.
                # Cache empty sentinel so we don't re-fetch within the 24h TTL.
                self._cache_write("transcript", ticker, "")
                return None
            result = content[:max_chars]
            self._cache_write("transcript", ticker, result)
            return result
        except FMPEndpointError:
            return None
        except Exception as exc:
            log.debug("get_earnings_transcript %s failed: %s", ticker, exc)
            return None
