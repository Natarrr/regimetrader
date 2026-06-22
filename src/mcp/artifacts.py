# Path: src/mcp/artifacts.py
"""Read-only access layer over the pipeline's JSON artifacts under logs/.

Pure, side-effect-free queries — no FMP, no scoring math, no network. Every
method degrades gracefully (None / empty) when an artifact is missing, so the
MCP server stays up even before the first pipeline run. The artifact shapes are
those emitted by run_pipeline / cook_toplists:

    intel_source_status.json  {source_meta, _edgar_meta, results:[{ticker,
                               sector, cap_tier, market_cap, *_score, ...}]}
    top_lists.json            {top_buys_usa|europe|asia:[{ticker, final_score,
                               badge, market, factors}], vix, vix_regime,
                               kill_switch, ...}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_INTEL = "intel_source_status.json"
_TOPLISTS = "top_lists.json"

_MARKET_BUCKETS = {
    "usa": "top_buys_usa",
    "europe": "top_buys_europe",
    "asia": "top_buys_asia",
}

# Small/mid sleeves: where INTL small-caps (eu/asia_mid_small) and the US SMID
# leverage sleeve (top_buys_smid) live after cook — not in top_buys_*. Surfaced
# so the MCP shows the full small/mid/large selection. Each carries its own
# `market`; top_buys_smid is US, so it defaults there.
_SLEEVE_BUCKETS = {
    "eu_mid_small": "EUROPE",
    "asia_mid_small": "ASIA",
    "top_buys_smid": "USA",
}


class ArtifactStore:
    """Queries over the committed pipeline artifacts in a logs directory."""

    def __init__(self, logs_dir: Path | str = "logs") -> None:
        self._dir = Path(logs_dir)

    # ── low-level ────────────────────────────────────────────────────────────

    def _load(self, name: str) -> Optional[Any]:
        path = self._dir / name
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _results(self) -> List[Dict[str, Any]]:
        doc = self._load(_INTEL)
        rows = (doc or {}).get("results") if isinstance(doc, dict) else None
        return rows if isinstance(rows, list) else []

    def _ranked_rows(self) -> List[Dict[str, Any]]:
        """All ranked rows across the large top-buys lists AND the small/mid
        sleeves, deduped by ticker (core buckets win). Each row is tagged with
        its source `bucket` and a `market` fallback so market filtering works."""
        doc = self._load(_TOPLISTS)
        if not isinstance(doc, dict):
            return []
        out: List[Dict[str, Any]] = []
        seen: set = set()
        buckets = list(_MARKET_BUCKETS.values()) + list(_SLEEVE_BUCKETS)
        for col in buckets:
            rows = doc.get(col)
            if not isinstance(rows, list):
                continue
            for r in rows:
                if not isinstance(r, dict):
                    continue
                key = str(r.get("ticker", "")).upper()
                if not key or key in seen:
                    continue
                seen.add(key)
                rr = dict(r)
                rr.setdefault("bucket", col)
                rr.setdefault("market", _SLEEVE_BUCKETS.get(col, ""))
                out.append(rr)
        return out

    # ── public queries ───────────────────────────────────────────────────────

    def ticker_score(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Per-ticker factor breakdown (intel_source_status) merged with the
        ranked final_score/badge (top_lists) when the name is on a top-list.
        None when the ticker is absent from the scored universe."""
        key = (ticker or "").strip().upper()
        if not key:
            return None
        row = next((r for r in self._results()
                    if str(r.get("ticker", "")).upper() == key), None)
        ranked = next((r for r in self._ranked_rows()
                       if str(r.get("ticker", "")).upper() == key), None)
        if row is None and ranked is None:
            return None

        out: Dict[str, Any] = dict(row or {"ticker": key})
        # Split raw *_score fields into a nested `factors` block for clarity.
        factors = {k: v for k, v in out.items() if k.endswith("_score")}
        for k in factors:
            out.pop(k, None)
        out["factors"] = factors
        if ranked is not None:
            out["final_score"] = ranked.get("final_score")
            out["badge"] = ranked.get("badge")
            out.setdefault("market", ranked.get("market"))
            for k, v in (ranked.get("factors") or {}).items():
                out["factors"].setdefault(k, v)
        return out

    def toplists(
        self, market: Optional[str] = None, badge: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ranked names (optionally filtered by market/badge) plus the regime."""
        rows = self._ranked_rows()
        if market:
            mk = market.strip().lower()
            rows = [r for r in rows if str(r.get("market", "")).lower() == mk]
        if badge:
            bd = badge.strip().lower()
            rows = [r for r in rows if str(r.get("badge", "")).lower() == bd]
        rows = sorted(rows, key=lambda r: r.get("final_score") or 0.0, reverse=True)
        return {"names": rows, "regime": self.regime()}

    def regime(self) -> Dict[str, Any]:
        """VIX safety-gate state from top_lists.json."""
        doc = self._load(_TOPLISTS)
        doc = doc if isinstance(doc, dict) else {}
        return {
            "vix": doc.get("vix"),
            "vix_regime": doc.get("vix_regime"),
            "kill_switch": doc.get("kill_switch"),
        }

    def source_health(self) -> Dict[str, Any]:
        """Data-source freshness + EDGAR run metadata."""
        doc = self._load(_INTEL)
        doc = doc if isinstance(doc, dict) else {}
        return {
            "source_meta": doc.get("source_meta", {}),
            "edgar_meta": doc.get("_edgar_meta", {}),
        }

    def search_universe(
        self,
        min_score: Optional[float] = None,
        market: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Ranked names filtered by score floor / market, score-descending."""
        rows = self.toplists(market=market)["names"]
        if min_score is not None:
            rows = [r for r in rows if (r.get("final_score") or 0.0) >= min_score]
        return rows
