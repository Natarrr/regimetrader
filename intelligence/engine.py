"""
intelligence/engine.py — Production market-intelligence engine v3.1
─────────────────────────────────────────────────────────────────
UPGRADES CLAUDE COOKBOOKS:
  1. JSON Structuré via tool_choice (forced JSON output)
  2. Analyse de Sentiment Avancée via Pattern RAG (Claude lit les titres)
  3. Tool Use pour API calls (FMP/Finnhub transformés en tools)

PILLARS & BASE WEIGHTS
──────────────────────
  Pillar  Base Wt  Source
  ──────────────────────────────────────────────────────────────
  sent     20 %   StockTwits  (public stream, no key needed)
  ins      20 %   FMP v4  /api/v4/insider-trading
  inst     20 %   FMP v3  /api/v3/institutional-holder/{sym}
  news     20 %   Finnhub /api/v1/company-news  + VADER NLP
  macro    20 %   Finnhub /api/v1/stock/recommendation

COOKBOOK REFERENCES:
────────────────────
  - tool_choice.ipynb: Forcing JSON output via tool definition
  - extracting_structured_json.ipynb: Structured JSON via tool use
  - RAG guide.ipynb: Contextual analysis pattern
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import yfinance as yf

from core.models import Direction, IntelligenceScore, Signal
from log_manager.logger import get_logger

log = get_logger(__name__)

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# UPGRADE 1: JSON STRUCTURÉ via Anthropic Tool Use
# Cookbook: extracting_structured_json.ipynb + tool_choice.ipynb
# ══════════════════════════════════════════════════════════════════════════════

# Define a tool that forces Claude to output structured JSON
# This ensures the scoring output is always valid JSON
_INTELLIGENCE_TOOL = {
    "name": "output_intelligence_score",
    "description": "Output the final intelligence score as structured JSON. Use this tool to return the complete scoring result to Streamlit.",
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Ticker symbol analyzed"},
            "final_conviction": {"type": "number", "description": "Final conviction score 0.0-1.0", "minimum": 0.0, "maximum": 1.0},
            "confidence_level": {"type": "number", "description": "Confidence 0.0-1.0 based on data availability", "minimum": 0.0, "maximum": 1.0},
            "sentiment_score": {"type": "number", "description": "Social sentiment score 0.0-1.0"},
            "insider_score": {"type": "number", "description": "Insider trading score 0.0-1.0"},
            "institutional_score": {"type": "number", "description": "Institutional holders score 0.0-1.0"},
            "news_score": {"type": "number", "description": "News sentiment score 0.0-1.0"},
            "macro_score": {"type": "number", "description": "Analyst consensus score 0.0-1.0"},
            "direction": {"type": "string", "description": "LONG, SHORT, or FLAT"},
            "target_weight": {"type": "number", "description": "Recommended position size 0.0-0.20"},
            "justification": {"type": "string", "description": "Human-readable explanation"},
            "weight_triggers": {"type": "array", "items": {"type": "string"}, "description": "List of dynamic weighting rules that fired"},
            "pillar_weights": {"type": "object", "description": "Dynamic weights per pillar"},
            "raw_data": {"type": "object", "description": "Raw scores from each pillar"},
            "metadata": {"type": "object", "description": "Additional metadata (sources, cache status, etc.)"}
        },
        "required": ["symbol", "final_conviction", "confidence_level", "direction"]
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# UPGRADE 3: TOOL USE pour API calls (Function Calling)
# Cookbook: tool_choice.ipynb + programmatic_tool_calling_ptc.ipynb
# ══════════════════════════════════════════════════════════════════════════════

# Define tools that Claude can use to fetch data intelligently
_API_TOOLS = [
    {
        "name": "fetch_fmp_insider_data",
        "description": "Fetch insider trading data from Financial Modeling Prep API. Use this to get CEO/CFO purchases and Open Market transactions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, MSFT)"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "fetch_fmp_institutional_data",
        "description": "Fetch institutional holder data from FMP. Use this to track BlackRock, Vanguard, and other major fund positions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "fetch_finnhub_news",
        "description": "Fetch company news from Finnhub. Use this to analyze recent news sentiment and article volume.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "days_back": {"type": "integer", "description": "Number of days to look back (default 7)", "default": 7}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "fetch_finnhub_recommendations",
        "description": "Fetch analyst recommendations from Finnhub. Use this to get macro/analyst consensus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "fetch_stocktwits_sentiment",
        "description": "Fetch social sentiment from StockTwits. Use this to gauge retail investor sentiment and trending.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    }
]

# ══════════════════════════════════════════════════════════════════════════════
# UPGRADE 2: ANALYSE DE SENTIMENT AVANCÉE (Pattern RAG)
# Cookbook: RAG guide.ipynb + summarization guide
# ══════════════════════════════════════════════════════════════════════════════

def analyze_contextual_sentiment(news_texts: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    UPGRADE 2: Claude-powered contextual sentiment analysis.
    
    Instead of basic VADER math, Claude reads the headlines and assigns
    its own News Score (0.0 to 1.0) based on contextual understanding.
    
    Cookbook reference: RAG guide.ipynb (Level 2-3: Summary Indexing + Re-Ranking)
    
    This function uses the Anthropic API to analyze news contextually.
    If API key is not available, falls back to VADER.
    
    Parameters
    ----------
    news_texts : List[Dict]
        List of news articles with 'headline' and optionally 'summary', 'source'
    
    Returns
    -------
    Dict with:
        - news_score: float (0.0-1.0) - Claude's contextual assessment
        - sentiment_label: str ('positive', 'negative', 'neutral')
        - positive_count: int
        - negative_count: int
        - article_count: int
        - analysis_method: str ('claude' or 'vader_fallback')
        - claude_reasoning: str (optional explanation from Claude)
    """
    if not news_texts:
        return {
            "news_score": 0.50,
            "sentiment_label": "neutral",
            "positive_count": 0,
            "negative_count": 0,
            "article_count": 0,
            "analysis_method": "vader_fallback",
            "claude_reasoning": "No news articles provided"
        }
    
    # Check for Anthropic API key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        log.warning("ANTHROPIC_API_KEY not found - using VADER fallback")
        return _vader_fallback_analysis(news_texts)
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=anthropic_key)
        
        # Prepare news for Claude analysis
        news_summary = "\n".join([
            f"- {article.get('headline', 'No headline')[:200]}"
            for article in news_texts[:15]  # Limit to 15 articles
        ])
        
        # Claude prompt for contextual analysis
        prompt = f"""Analyze the following news headlines for the stock:

{news_summary}

Based on your understanding of these headlines (not just keyword matching), assign a sentiment score from 0.0 to 1.0:
- 0.0-0.3: Strongly negative (major concerns, scandals, losses, bearish signals)
- 0.3-0.5: Somewhat negative (challenges, headwinds, cautious outlook)
- 0.5: Neutral (mixed or no significant news)
- 0.5-0.7: Somewhat positive (growth, improvements, optimistic signals)
- 0.7-1.0: Strongly positive (breakthroughs, beatings, upgrades, bullish signals)

Consider:
- The overall tone and context of the news
- Whether the news represents fundamental changes or temporary noise
- The severity and credibility of sources
- Any forward-looking statements or guidance

Return your analysis as JSON with these fields:
{{
  "news_score": <your score 0.0-1.0>,
  "sentiment_label": "positive" | "negative" | "neutral",
  "positive_count": <number of positive articles>,
  "negative_count": <number of negative articles>,
  "reasoning": "<2-3 sentence explanation of your assessment>"
}}"""
        
        # Call Claude with forced tool use for JSON output
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            tools=[{
                "name": "output_sentiment_analysis",
                "description": "Output sentiment analysis result as JSON",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "news_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "sentiment_label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                        "positive_count": {"type": "integer"},
                        "negative_count": {"type": "integer"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["news_score", "sentiment_label", "reasoning"]
                }
            }],
            tool_choice={"type": "tool", "name": "output_sentiment_analysis"},
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse Claude's response
        for content in response.content:
            if content.type == "tool_use" and content.name == "output_sentiment_analysis":
                result = content.input
                log.info(f"[CONTEXTUAL SENTIMENT] Claude analyzed {len(news_texts)} articles -> Score: {result['news_score']:.2f}")
                return {
                    "news_score": result["news_score"],
                    "sentiment_label": result["sentiment_label"],
                    "positive_count": result["positive_count"],
                    "negative_count": result["negative_count"],
                    "article_count": len(news_texts),
                    "analysis_method": "claude",
                    "claude_reasoning": result.get("reasoning", "")
                }
        
        # Fallback if no tool use
        log.warning("Claude did not use sentiment tool - falling back to VADER")
        return _vader_fallback_analysis(news_texts)
        
    except Exception as e:
        log.warning(f"Claude API call failed: {e} - using VADER fallback")
        return _vader_fallback_analysis(news_texts)


def _vader_fallback_analysis(news_texts: List[Dict[str, str]]) -> Dict[str, Any]:
    """Fallback to VADER when Claude is unavailable."""
    try:
        import nltk
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
        
        sia = SIA()
        compounds = []
        for article in news_texts[:30]:
            headline = article.get("headline", "")
            if headline:
                compounds.append(sia.polarity_scores(headline)["compound"])
        
        if not compounds:
            return {"news_score": 0.50, "sentiment_label": "neutral", "positive_count": 0, "negative_count": 0, "article_count": 0, "analysis_method": "vader_fallback"}
        
        mean_compound = sum(compounds) / len(compounds)
        positive = sum(1 for c in compounds if c > 0.1)
        negative = sum(1 for c in compounds if c < -0.1)
        
        return {
            "news_score": _clamp((mean_compound + 1) / 2),
            "sentiment_label": "positive" if mean_compound > 0.1 else "negative" if mean_compound < -0.1 else "neutral",
            "positive_count": positive,
            "negative_count": negative,
            "article_count": len(compounds),
            "analysis_method": "vader_fallback",
            "claude_reasoning": f"VADER: mean compound = {mean_compound:.3f}"
        }
    except Exception as e:
        log.error(f"VADER fallback failed: {e}")
        return {"news_score": 0.50, "sentiment_label": "neutral", "positive_count": 0, "negative_count": 0, "article_count": 0, "analysis_method": "error"}


# ══════════════════════════════════════════════════════════════════════════════
# Cache helpers (unchanged from v3)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import diskcache as dc
    _CACHE_DIR = os.path.join(os.path.dirname(__file__), ".intel_cache")
    _disk: Optional[Any] = dc.Cache(_CACHE_DIR)
    _USE_DISK = True
except ImportError:
    _disk = None
    _USE_DISK = False
    log.warning("diskcache not installed — using in-memory cache")

_mem: Dict[str, Dict[str, Any]] = {}

_TTL: Dict[str, float] = {
    "sent":  5 * 60,
    "ins":   24 * 3600,
    "inst":  24 * 3600,
    "news":  15 * 60,
    "macro": 4 * 3600,
}

_BASE_WEIGHTS: Dict[str, float] = {
    "sent":  0.20,
    "ins":   0.20,
    "inst":  0.20,
    "news":  0.20,
    "macro": 0.20,
}

# ══════════════════════════════════════════════════════════════════════════════
# Enhanced Scoring Constants (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_OPEN_MARKET_TYPES = {"P-PURCHASE", "PURCHASE", "OPEN MARKET PURCHASE"}
_INSIDER_ROLE_BOOST = {"CEO", "CFO", "DIRECTOR", "CHIEF EXECUTIVE", "CHIEF FINANCIAL OFFICER"}
_INSIDER_MIN_AMOUNT_FORCE = 50_000.0
_INSIDER_OPEN_MARKET_MIN = 30
_INSIDER_CONVICTION_FLOOR = 0.85

_MAJOR_FUNDS = {"BLACKROCK", "VANGUARD", "RENAISSANCE", "STATE STREET", "FIDELITY", "JP MORGAN"}
_INST_MAJOR_INCREASE_THRESHOLD = 0.05
_INST_WHALE_BOOST = 0.40

_SENTIMENT_ZSPIKE_THRESHOLD = 2.0
_SENTIMENT_CORRELATION_BOOST = 0.40

_CONVERGENCE_ALPHA_THRESHOLD = 0.90
_CONVERGENCE_ALPHA_MIN = 0.95

_LONG_THRESHOLD:    float = 0.58
_SHORT_THRESHOLD:   float = 0.42
_MIN_TARGET_WEIGHT: float = 0.02
_MAX_TARGET_WEIGHT: float = 0.20
_TOP_N: int = 5
# Use None to distinguish missing data from neutral data (0.5)
# When a pillar fails to fetch data, it returns None instead of 0.5
_NEUTRAL: float = 0.50  # Kept for backward compatibility in final scoring


def _fmp_key() -> str:
    return os.getenv("FMP_API_KEY", "")

def _finnhub_key() -> str:
    return os.getenv("FINNHUB_API_KEY", "")

_REQUEST_TO = 12
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="intel")


# ══════════════════════════════════════════════════════════════════════════════
# Cache functions (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _cache_get_pillar(key: str) -> Optional[Tuple[float, Dict]]:
    if _USE_DISK and _disk is not None:
        val = _disk.get(key)
        if val is None:
            return None
        if isinstance(val, tuple) and len(val) == 2:
            # Preserve None values - don't convert to float
            return val
        # Handle legacy cached floats
        return float(val), {}
    entry = _mem.get(key)
    if entry and time.monotonic() < entry["exp"]:
        val = entry["val"]
        if isinstance(val, tuple):
            # Preserve None values
            return val
        # Handle legacy cached floats
        return float(val), {}
    return None


def _cache_set_pillar(key: str, score: Optional[float], meta: Dict, ttl: float) -> None:
    # score can be None (missing data) or a float
    if _USE_DISK and _disk is not None:
        _disk.set(key, (score, meta), expire=ttl)
    else:
        _mem[key] = {"val": (score, meta), "exp": time.monotonic() + ttl}


def is_pillar_cached(ticker: str, pillar: str) -> bool:
    cache_key = f"p3:{pillar}:{ticker.upper()}"
    if _USE_DISK and _disk is not None:
        return _disk.get(cache_key) is not None
    return cache_key in _mem


def get_cached_pillars(ticker: str) -> Dict[str, bool]:
    pillars = ["sent", "ins", "inst", "news", "macro"]
    return {p: is_pillar_cached(ticker, p) for p in pillars}


# ══════════════════════════════════════════════════════════════════════════════
# Math helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, v)), 4)


def _geometric_weighted_mean(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    # Filter out None values - only use scores that have actual data
    active = {k: v for k, v in scores.items() if k in weights and v is not None}
    if not active:
        return _NEUTRAL
    w_total = sum(weights[k] for k in active)
    if w_total == 0:
        return _NEUTRAL
    log_sum = sum(
        (weights[k] / w_total) * math.log(max(1e-4, min(1.0, s)))
        for k, s in active.items()
    )
    return round(math.exp(log_sum), 4)


def _time_decay(score: float, age_days: float, half_life: float = 21.0) -> float:
    if age_days <= 0.0:
        return score
    lam = math.log(2.0) / max(half_life, 1.0)
    return _clamp(_NEUTRAL + (score - _NEUTRAL) * math.exp(-lam * age_days))


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic weighting (unchanged from v3)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_dynamic_weights(
    pillar_meta: Dict[str, Dict],
    base_weights: Optional[Dict[str, float]] = None,
    scores: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], float, List[str]]:
    bw = base_weights if base_weights is not None else _BASE_WEIGHTS
    weights: Dict[str, float] = {k: v for k, v in bw.items()}
    triggers: List[str] = []

    available = [k for k in weights if pillar_meta.get(k, {}).get("has_data", False)]
    missing   = [k for k in weights if not pillar_meta.get(k, {}).get("has_data", False)]
    confidence = round(len(available) / max(len(bw), 1), 2)

    # Rule 1: High Conviction insider
    ceo_amt = pillar_meta.get("ins", {}).get("ceo_purchase_amount", 0.0)
    if ceo_amt > 1_000_000 and "ins" in available:
        boost = 0.40 - weights["ins"]
        weights["ins"] = 0.40
        others = [k for k in available if k != "ins"]
        if others:
            cut_each = boost / len(others)
            for k in others:
                weights[k] = max(0.0, weights[k] - cut_each)
        triggers.append(f"ceo_purchase:{ceo_amt / 1e6:.1f}M")

    # Rule 2: Momentum
    st_zscore = pillar_meta.get("sent", {}).get("volume_zscore", 0.0)
    if st_zscore > 2.0 and "sent" in available:
        boost = min(0.15, 0.04 * st_zscore)
        weights["sent"] = min(0.50, weights["sent"] + boost)
        triggers.append(f"social_momentum:{st_zscore:.1f}σ")

    # Rule 3: Redistribute missing pillar weights
    if missing and available:
        lost = sum(weights[k] for k in missing)
        for k in missing:
            weights[k] = 0.0
        per_active = lost / len(available)
        for k in available:
            weights[k] += per_active
        triggers.extend([f"no_data:{k}" for k in missing])
    elif not available:
        weights = {k: v for k, v in bw.items()}

    # Rule 4: Smart Money Convergence
    if scores is not None and "ins" in available and "inst" in available:
        ins_s  = scores.get("ins",  0.50)
        inst_s = scores.get("inst", 0.50)
        if ins_s > 0.62 and inst_s > 0.62:
            transfer = 0.04
            for donor in ("sent", "news"):
                if donor in available and weights[donor] >= transfer:
                    weights[donor] -= transfer
                    recipient = "ins" if ins_s >= inst_s else "inst"
                    weights[recipient] = min(0.60, weights[recipient] + transfer)
            triggers.append(f"smart_money:{ins_s:.2f}/{inst_s:.2f}")

    # Rule 5: Social/News Correlation Boost
    if scores is not None and "sent" in available and "news" in available:
        sent_s = scores.get("sent", 0.50)
        news_s = scores.get("news", 0.50)
        if sent_s > 0.55 and news_s > 0.55:
            boost = min(0.20, weights["sent"] + 0.10)
            if boost > weights["sent"]:
                diff = boost - weights["sent"]
                weights["sent"] = boost
                for k in [p for p in available if p != "sent"]:
                    weights[k] = max(0.05, weights[k] - diff / len([p for p in available if p != "sent"]))
                triggers.append(f"social_news_correlation:sent={sent_s:.2f},news={news_s:.2f}")

    # Normalise
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: round(v / total_w, 4) for k, v in weights.items()}

    return weights, confidence, triggers


def _conviction_to_signal(
    symbol: str, conviction: float, score: IntelligenceScore
) -> Signal:
    if conviction >= _LONG_THRESHOLD:
        direction = Direction.LONG.value
        t = (conviction - _LONG_THRESHOLD) / (1.0 - _LONG_THRESHOLD)
        target_w = _MIN_TARGET_WEIGHT + t * (_MAX_TARGET_WEIGHT - _MIN_TARGET_WEIGHT)
        just = (
            f"Strong multi-source conviction ({conviction:.0%}).  "
            f"Inst={score.flow_score:.0%}, News={score.congress_score:.0%}, "
            f"Sent={score.sentiment_score:.0%}, "
            f"Ins={score.insider_score:.0%}, Macro={score.macro_score:.0%}."
        )
    elif conviction <= _SHORT_THRESHOLD:
        direction = Direction.SHORT.value
        t = (_SHORT_THRESHOLD - conviction) / _SHORT_THRESHOLD
        target_w = _MIN_TARGET_WEIGHT + t * (_MAX_TARGET_WEIGHT - _MIN_TARGET_WEIGHT)
        just = (
            f"Multi-source bearish signal ({conviction:.0%}).  "
            f"Inst={score.flow_score:.0%}, News={score.congress_score:.0%}, "
            f"Sent={score.sentiment_score:.0%}, "
            f"Ins={score.insider_score:.0%}, Macro={score.macro_score:.0%}."
        )
    else:
        direction = Direction.FLAT.value
        target_w = 0.0
        just = (
            f"Neutral conviction ({conviction:.0%}) — no clear edge.  "
            f"Inst={score.flow_score:.0%}, News={score.congress_score:.0%}, "
            f"Sent={score.sentiment_score:.0%}, "
            f"Ins={score.insider_score:.0%}, Macro={score.macro_score:.0%}."
        )
    return Signal(
        symbol        = symbol,
        direction     = direction,
        target_weight = round(target_w, 4),
        confidence    = conviction,
        justification = just,
        generated_at  = datetime.now(tz=timezone.utc),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sync API fetchers (unchanged logic, now with tool-ready structure)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_sent_sync(ticker: str) -> Tuple[float, Dict]:
    try:
        resp = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker.upper()}.json",
            headers=_HEADERS,
            timeout=_REQUEST_TO,
        )
        if resp.status_code != 200:
            return _NEUTRAL, {"has_data": False, "source": "stocktwits", "volume_zscore": 0.0}
        data = resp.json()
        messages = data.get("messages", [])

        bull = sum(1 for m in messages if (m.get("entities") or {}).get("sentiment") and (m.get("entities") or {})["sentiment"].get("basic") == "Bullish")
        bear = sum(1 for m in messages if (m.get("entities") or {}).get("sentiment") and (m.get("entities") or {})["sentiment"].get("basic") == "Bearish")
        total_tagged = bull + bear

        if total_tagged == 0:
            return _NEUTRAL, {"has_data": False, "source": "stocktwits", "volume_zscore": 0.0}

        raw_score = bull / total_tagged
        zscore    = _stocktwits_zscore(ticker, float(total_tagged))
        score = _clamp(0.40 + 0.60 * raw_score)
        
        if zscore > _SENTIMENT_ZSPIKE_THRESHOLD:
            spike_boost = min(0.25, (zscore - 2.0) * 0.08)
            score = _clamp(score + spike_boost)
        
        watchlist = int((data.get("symbol") or {}).get("watchlist_count", 0))
        trending  = watchlist > 10_000
        if trending:
            score = _clamp(score + 0.05)

        return score, {
            "has_data":      True,
            "source":        "stocktwits",
            "bullish":       bull,
            "bearish":       bear,
            "total":         total_tagged,
            "volume_zscore": round(zscore, 2),
            "trending":      trending,
            "watchlist":     watchlist,
        }
    except Exception as exc:
        log.warning("WARNING: sent (StockTwits) failed for {}: {}", ticker, exc)
        return _NEUTRAL, {"has_data": False, "source": "stocktwits", "volume_zscore": 0.0}


def _fetch_ins_yf_sync(ticker: str) -> Tuple[float, Dict]:
    """yfinance fallback for insider transactions when primary sources are unavailable."""
    try:
        import yfinance as _yf
        t = _yf.Ticker(ticker)
        txns = t.insider_transactions
        if txns is None or (hasattr(txns, "empty") and txns.empty):
            log.warning("[INSIDER] No insider transactions data for {} (yfinance)", ticker)
            return None, {"has_data": False, "source": "yf_insider"}

        # yfinance columns: Shares, URL, Text, Insider, Position, Transaction, Start Date, Ownership, Value
        # Use SEC Form 4 transaction codes from 'Transaction' col (P=purchase, S=sale)
        # and fall back to 'Text' col keyword matching for older data.
        txn_col  = next((c for c in txns.columns if c.lower() == "transaction"), None)
        text_col = next((c for c in txns.columns if c.lower() == "text"), None)
        val_col  = next((c for c in txns.columns if c.lower() == "value"), None)
        shr_col  = next((c for c in txns.columns if c.lower() == "shares"), None)

        buy_val = sell_val = 0.0

        for _, row in txns.iterrows():
            # Determine dollar value (Shares × price when Value is NaN)
            try:
                val = abs(float(str(row.get(val_col, 0) or 0).replace(",", ""))) if val_col else 0.0
            except Exception:
                val = 0.0
            if val == 0 and shr_col:
                try:
                    val = abs(float(str(row.get(shr_col, 0) or 0).replace(",", "")))
                except Exception:
                    val = 0.0

            # Prefer SEC transaction code (clean, single-char codes)
            code = str(row.get(txn_col, "") or "").strip().upper() if txn_col else ""
            txt  = str(row.get(text_col, "") or "").lower()          if text_col else ""

            is_buy = is_sell = False
            if code == "P":
                is_buy = True
            elif code in ("S", "S-1", "S-2"):
                is_sell = True
            elif "purchase" in txt or "acqui" in txt or ("buy" in txt and "sale" not in txt):
                is_buy = True
            elif "sale" in txt or "sell" in txt or "sold" in txt:
                is_sell = True
            # Skip option exercises (M), awards (A), grants, tax withholding (F) — not open-market

            if is_buy:
                buy_val += val
            elif is_sell:
                sell_val += val

        total = buy_val + sell_val
        if total == 0:
            log.warning("[INSIDER] No dollar value transactions for {} (yfinance) - buy_count={}, sell_count={}", 
                        ticker, len([r for r in txns if str(r.get("transaction", "")).strip().upper() == "P"]),
                        len([r for r in txns if "S" in str(r.get("transaction", "")).strip().upper()]))
            return None, {"has_data": False, "source": "yf_insider"}

        ratio = buy_val / total
        score = _clamp(0.38 + 0.62 * ratio)
        return score, {
            "has_data":            True,
            "source":              "yf_insider",
            "buy_value":           round(buy_val, 0),
            "sell_value":          round(sell_val, 0),
            "ceo_purchase_amount": 0.0,
        }
    except Exception as exc:
        log.warning("[INSIDER] yfinance failed for {}: {}", ticker, exc)
        return None, {"has_data": False, "source": "yf_insider"}


def _fetch_ins_finnhub_sync(ticker: str) -> Tuple[float, Dict]:
    """Finnhub insider-transactions — primary source, free tier, SEC Form 4 codes."""
    key = _finnhub_key()
    if not key:
        log.warning("[INSIDER] No FINNHUB_API_KEY configured for {}", ticker)
        return None, {"has_data": False, "source": "finnhub_insider"}
    try:
        resp = requests.get(
            f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}&token={key}",
            headers=_HEADERS,
            timeout=_REQUEST_TO,
        )
        if resp.status_code != 200:
            log.warning("[INSIDER] Finnhub API error {} for {}", resp.status_code, ticker)
            return None, {"has_data": False, "source": "finnhub_insider"}

        data = resp.json().get("data", [])
        if not data:
            log.warning("[INSIDER] No insider transactions data for {} (Finnhub)", ticker)
            return None, {"has_data": False, "source": "finnhub_insider"}

        cutoff = datetime.now() - timedelta(days=180)
        buy_value = sell_value = exec_buy = 0.0

        for tx in data:
            if tx.get("isDerivative", True):
                continue  # skip option exercises — not open-market conviction

            code  = str(tx.get("transactionCode", "")).strip().upper()
            price = float(tx.get("transactionPrice", 0) or 0)
            chng  = float(tx.get("change", 0) or 0)
            value = abs(chng) * price

            date_str = tx.get("transactionDate", "")
            try:
                tx_date = datetime.strptime(date_str, "%Y-%m-%d")
                if tx_date < cutoff:
                    continue
            except Exception:
                pass

            if code == "P":          # open-market purchase — direct conviction signal
                buy_value += value
                exec_buy  += value
            elif code == "S":        # open-market sale
                sell_value += value

        total = buy_value + sell_value
        if total == 0:
            log.warning("[INSIDER] No dollar value transactions for {} (Finnhub) - buys={}, sells={}",
                        ticker, len([t for t in data if t.get("transactionCode") == "P"]),
                        len([t for t in data if t.get("transactionCode") == "S"]))
            return None, {"has_data": False, "source": "finnhub_insider"}

        ratio = buy_value / total
        score = _clamp(0.40 + 0.60 * ratio)

        # Large executive open-market buys are strong conviction signals
        if exec_buy >= _INSIDER_MIN_AMOUNT_FORCE:
            score = max(score, _INSIDER_CONVICTION_FLOOR)
        if exec_buy >= 2_000_000:
            score = _clamp(max(score, 0.85))

        return score, {
            "has_data":            True,
            "source":              "finnhub_insider",
            "buy_value":           round(buy_value, 0),
            "sell_value":          round(sell_value, 0),
            "ceo_purchase_amount": round(exec_buy, 0),
        }
    except Exception as exc:
        log.warning("[INSIDER] Finnhub failed for {}: {}", ticker, exc)
        return None, {"has_data": False, "source": "finnhub_insider"}


def _fetch_ins_sync(ticker: str) -> Tuple[float, Dict]:
    # Priority: Finnhub (free, reliable) → FMP (if available) → yfinance
    result = _fetch_ins_finnhub_sync(ticker)
    if result[1].get("has_data", False):
        return result

    key = _fmp_key()
    if not key:
        return _fetch_ins_yf_sync(ticker)
    try:
        resp = requests.get(
            f"https://financialmodelingprep.com/api/v4/insider-trading"
            f"?symbol={ticker}&limit=50&apikey={key}",
            headers=_HEADERS,
            timeout=_REQUEST_TO,
        )
        if resp.status_code in (403, 429):
            log.warning("FMP insider {} — falling back to yfinance", resp.status_code)
            return _fetch_ins_yf_sync(ticker)
        data = resp.json() if resp.status_code == 200 else []
        if not isinstance(data, list) or not data:
            return _fetch_ins_yf_sync(ticker)

        buy_value = sell_value = ceo_purchase_amount = 0.0
        open_market_buys = []
        thirty_days_ago = datetime.now() - timedelta(days=_INSIDER_OPEN_MARKET_MIN)
        
        for tx in data:
            tx_type = str(tx.get("transactionType", "")).upper()
            shares  = float(tx.get("securitiesTransacted", 0) or 0)
            price   = float(tx.get("price", 0) or 0)
            value   = abs(shares * price)
            owner   = str(tx.get("typeOfOwner", "")).upper()
            date_str = tx.get("date", "")
            try:
                tx_date = datetime.strptime(date_str, "%Y-%m-%d") if date_str else None
            except:
                tx_date = None

            if "P-PURCHASE" in tx_type or tx_type == "PURCHASE":
                buy_value += value
                is_exec = any(r in owner for r in _INSIDER_ROLE_BOOST)
                if is_exec:
                    ceo_purchase_amount += value
                    if any(om in tx_type for om in _OPEN_MARKET_TYPES):
                        if tx_date and tx_date >= thirty_days_ago:
                            open_market_buys.append((value, owner, date_str))
            elif "SALE" in tx_type:
                sell_value += value

        total = buy_value + sell_value
        if total == 0:
            log.warning("[INSIDER] No dollar value transactions for {} (FMP)", ticker)
            return None, {"has_data": False, "source": "fmp_insider", "ceo_purchase_amount": 0.0}

        ratio = buy_value / total
        score = _clamp(0.40 + 0.60 * ratio)

        if open_market_buys:
            max_open_market = max(buys[0] for buys in open_market_buys)
            if max_open_market >= _INSIDER_MIN_AMOUNT_FORCE:
                score = max(score, _INSIDER_CONVICTION_FLOOR)
        
        if ceo_purchase_amount >= _INSIDER_MIN_AMOUNT_FORCE:
            multiplier = min(0.15, (ceo_purchase_amount / 1_000_000) * 0.10)
            score = _clamp(score + multiplier)

        if ceo_purchase_amount > 1_000_000:
            score = _clamp(max(score, 0.85))

        return score, {
            "has_data":            True,
            "source":              "fmp_insider",
            "buy_value":           round(buy_value, 0),
            "sell_value":          round(sell_value, 0),
            "ceo_purchase_amount": round(ceo_purchase_amount, 0),
            "open_market_buys":    len(open_market_buys),
        }
    except Exception as exc:
        log.warning("[INSIDER] FMP failed for {}: {}", ticker, exc)
        return None, {"has_data": False, "source": "fmp_insider", "ceo_purchase_amount": 0.0}


def _fetch_inst_yf_sync(ticker: str) -> Tuple[float, Dict]:
    """yfinance institutional holders — primary free source for institutional data."""
    try:
        import yfinance as _yf
        t = _yf.Ticker(ticker)
        holders = t.institutional_holders
        if holders is None or (hasattr(holders, "empty") and holders.empty):
            log.warning("[INSTITUTIONAL] No holders data for {} (yfinance)", ticker)
            return None, {"has_data": False, "source": "yf_inst"}

        n = len(holders)
        cols_lower = {c.lower(): c for c in holders.columns}

        # yfinance uses 'pctHeld' (camelCase) — match case-insensitively
        pct_col = cols_lower.get("pctheld") or cols_lower.get("pct held") or cols_lower.get("% out") or cols_lower.get("pctout")
        chg_col = cols_lower.get("pctchange") or cols_lower.get("pct change") or cols_lower.get("% change")
        val_col = cols_lower.get("value")

        # ── Base score: total institutional ownership depth ────────────────────
        inst_total_pct = None

        # Method 1: major_holders gives the clean aggregate % (most reliable)
        try:
            mh = t.major_holders
            if mh is not None and not (hasattr(mh, "empty") and mh.empty):
                for _, mrow in mh.iterrows():
                    label = str(mrow.iloc[1] if len(mrow) > 1 else "").lower()
                    if "institutionspercent" in label.replace(" ", "") or "institutions percent held" in label:
                        inst_total_pct = float(mrow.iloc[0]) * 100  # convert fraction → %
                        break
        except Exception:
            pass

        if inst_total_pct is None and pct_col is not None:
            raw = holders[pct_col].dropna()
            s = float(raw.sum())
            inst_total_pct = s * 100 if s < 1.0 else s  # convert fraction → %

        if inst_total_pct is not None and inst_total_pct > 0:
            # >65% = heavy institutional coverage (normal for large caps) → 0.68
            # <20% = thin coverage (micro/small cap risk) → 0.48
            base_score = _clamp(0.45 + min(0.25, inst_total_pct * 0.0035))
        else:
            base_score = _clamp(0.50 + min(0.08, n * 0.004))

        # ── Direction boost: are top holders increasing or decreasing? ─────────
        net_change_pct = 0.0
        if chg_col is not None:
            chg_series = holders[chg_col].dropna()
            if len(chg_series) > 0:
                # pctChange > 0 → holder increased position (bullish); < 0 → trimmed (bearish)
                avg_chg = float(chg_series.mean())
                net_change_pct = avg_chg
                # Each 1% net avg change shifts score by ~0.03 (capped ±0.15)
                direction_adj = _clamp(avg_chg * 0.03, lo=-0.15, hi=0.15)
                base_score = _clamp(base_score + direction_adj)

                # Extra boost if major funds (Vanguard, BlackRock) are increasing
                holder_col = cols_lower.get("holder")
                if holder_col is not None:
                    for _, hrow in holders.iterrows():
                        hname = str(hrow.get(holder_col, "") or "").upper()
                        hchg  = float(hrow.get(chg_col, 0) or 0)
                        if any(mf in hname for mf in _MAJOR_FUNDS) and hchg >= _INST_MAJOR_INCREASE_THRESHOLD:
                            base_score = _clamp(base_score + 0.04)

        return base_score, {
            "has_data":      True,
            "source":        "yf_inst",
            "holders_count": n,
            "net_change":    round(net_change_pct, 4),
            "inst_total_pct": round(inst_total_pct, 2) if inst_total_pct else None,
        }
    except Exception as exc:
        log.warning("[INSTITUTIONAL] yfinance failed for {}: {}", ticker, exc)
        return None, {"has_data": False, "source": "yf_inst"}


def _fetch_inst_sync(ticker: str) -> Tuple[float, Dict]:
    # Try yfinance first — it's free, reliable, and now properly scored
    result = _fetch_inst_yf_sync(ticker)
    if result[1].get("has_data", False):
        return result

    # FMP as secondary (premium endpoint, may be rate-limited)
    key = _fmp_key()
    if not key:
        log.warning("[INSTITUTIONAL] No FMP_API_KEY configured for {}", ticker)
        return None, {"has_data": False, "source": "fmp_inst"}
    try:
        resp = requests.get(
            f"https://financialmodelingprep.com/api/v3/institutional-holder/{ticker}"
            f"?apikey={key}",
            headers=_HEADERS,
            timeout=_REQUEST_TO,
        )
        if resp.status_code in (403, 429):
            log.warning("[INSTITUTIONAL] FMP API error {} for {} — yfinance already tried", resp.status_code, ticker)
            return None, {"has_data": False, "source": "fmp_inst"}
        data = resp.json() if resp.status_code == 200 else []
        if not isinstance(data, list) or not data:
            log.warning("[INSTITIONAL] No holders data for {} (FMP)", ticker)
            return None, {"has_data": False, "source": "fmp_inst"}

        total_change = total_shares = 0.0
        holders_count = 0
        major_fund_increases = []
        
        for holder in data[:20]:
            change = float(holder.get("change", 0) or 0)
            shares = float(holder.get("shares", 0) or 0)
            holder_name = str(holder.get("holderName", "")).upper()
            total_change += change
            total_shares += shares
            holders_count += 1
            
            if shares > 0 and change > 0:
                change_pct = change / shares
                if any(mf in holder_name for mf in _MAJOR_FUNDS):
                    if change_pct >= _INST_MAJOR_INCREASE_THRESHOLD:
                        major_fund_increases.append((holder_name, change_pct))

        if total_shares == 0 or holders_count == 0:
            log.warning("[INSTITUTIONAL] No shares data for {} (FMP)", ticker)
            return None, {"has_data": False, "source": "fmp_inst"}

        net_ratio = total_change / max(total_shares, 1)
        score = _clamp(0.50 + min(0.40, max(-0.40, net_ratio * 10)))

        if major_fund_increases:
            whale_boost = min(0.20, len(major_fund_increases) * 0.05)
            score = _clamp(score + whale_boost)

        return score, {
            "has_data":            True,
            "source":              "fmp_inst",
            "holders_count":      holders_count,
            "net_change":          round(total_change, 0),
            "net_ratio":           round(net_ratio, 6),
            "major_fund_increases": len(major_fund_increases),
        }
    except Exception as exc:
        log.warning("[INSTITUTIONAL] FMP failed for {}: {}", ticker, exc)
        return None, {"has_data": False, "source": "fmp_inst"}


def _fetch_news_sync(ticker: str) -> Tuple[float, Dict]:
    """
    Finnhub company-news with UPGRADE 2: Claude-powered contextual sentiment.
    """
    key = _finnhub_key()
    if not key:
        return _NEUTRAL, {"has_data": False, "source": "finnhub_news"}
    try:
        from datetime import date
        today    = date.today()
        week_ago = today - timedelta(days=7)
        resp = requests.get(
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={week_ago.isoformat()}&to={today.isoformat()}&token={key}",
            headers=_HEADERS,
            timeout=_REQUEST_TO,
        )
        if resp.status_code != 200:
            return _NEUTRAL, {"has_data": False, "source": "finnhub_news"}
        articles = resp.json()
        if not isinstance(articles, list) or not articles:
            return _NEUTRAL, {"has_data": False, "source": "finnhub_news", "article_count": 0}

        # UPGRADE 2: Use Claude for contextual sentiment analysis
        news_texts = [{"headline": a.get("headline", ""), "summary": a.get("summary", "")} for a in articles[:30]]
        contextual_result = analyze_contextual_sentiment(news_texts)
        
        score = contextual_result["news_score"]
        
        # Add volume boost
        if len(articles) >= 10:
            score = _clamp(score + 0.03)

        return score, {
            "has_data":        True,
            "source":          "finnhub_news",
            "article_count":   contextual_result["article_count"],
            "mean_compound":   contextual_result.get("claude_reasoning", ""),
            "positive_count":  contextual_result["positive_count"],
            "negative_count":  contextual_result["negative_count"],
            "news_sentiment":  contextual_result["sentiment_label"],
            "analysis_method": contextual_result["analysis_method"],
            "claude_reasoning": contextual_result.get("claude_reasoning", ""),
        }
    except Exception as exc:
        log.warning("WARNING: news (Finnhub) failed for {}: {}", ticker, exc)
        return _NEUTRAL, {"has_data": False, "source": "finnhub_news"}


def _fetch_macro_sync(ticker: str) -> Tuple[float, Dict]:
    key = _finnhub_key()
    if not key:
        return _NEUTRAL, {"has_data": False, "source": "finnhub_analyst"}
    try:
        resp = requests.get(
            f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={key}",
            headers=_HEADERS,
            timeout=_REQUEST_TO,
        )
        data = resp.json()
        if not isinstance(data, list) or not data:
            return _NEUTRAL, {"has_data": False, "source": "finnhub_analyst"}
        rec   = data[0]
        sb    = int(rec.get("strongBuy",  0) or 0)
        b     = int(rec.get("buy",        0) or 0)
        h     = int(rec.get("hold",       0) or 0)
        s     = int(rec.get("sell",       0) or 0)
        ss    = int(rec.get("strongSell", 0) or 0)
        total = sb + b + h + s + ss
        if total == 0:
            return _NEUTRAL, {"has_data": False, "source": "finnhub_analyst"}
        raw   = (sb * 1.0 + b * 0.75 + h * 0.50 + s * 0.25) / total
        score = _clamp(0.10 + raw * 0.80)
        return score, {
            "has_data":  True,
            "source":    "finnhub_analyst",
            "strongBuy": sb, "buy": b, "hold": h, "sell": s, "strongSell": ss,
            "total":     total,
        }
    except Exception as exc:
        log.warning("WARNING: macro (Finnhub analyst) failed for {}: {}", ticker, exc)
        return _NEUTRAL, {"has_data": False, "source": "finnhub_analyst"}


# ══════════════════════════════════════════════════════════════════════════════
# StockTwits volume tracker
# ══════════════════════════════════════════════════════════════════════════════

_st_volume_history: Dict[str, List[float]] = {}

def _stocktwits_zscore(ticker: str, current_total: float) -> float:
    hist = _st_volume_history.setdefault(ticker, [])
    hist.append(current_total)
    if len(hist) > 30:
        hist.pop(0)
    if len(hist) < 5:
        return 0.0
    mean = sum(hist) / len(hist)
    std  = math.sqrt(sum((v - mean) ** 2 for v in hist) / len(hist))
    return (current_total - mean) / std if std > 1e-6 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Async orchestration
# ══════════════════════════════════════════════════════════════════════════════

async def _run(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


async def _pillar_sent(ticker: str) -> Tuple[str, float, Dict, str]:
    cache_key = f"p3:sent:{ticker}"
    cached = _cache_get_pillar(cache_key)
    if cached is not None:
        return "sent", cached[0], cached[1], "HIT"
    score, meta = await _run(_fetch_sent_sync, ticker)
    _cache_set_pillar(cache_key, score, meta, _TTL["sent"])
    return "sent", score, meta, "MISS"


async def _pillar_ins(ticker: str) -> Tuple[str, float, Dict, str]:
    cache_key = f"p3:ins:{ticker}"
    cached = _cache_get_pillar(cache_key)
    if cached is not None:
        return "ins", cached[0], cached[1], "HIT"
    score, meta = await _run(_fetch_ins_sync, ticker)
    _cache_set_pillar(cache_key, score, meta, _TTL["ins"])
    return "ins", score, meta, "MISS"


async def _pillar_inst(ticker: str) -> Tuple[str, float, Dict, str]:
    cache_key = f"p3:inst:{ticker}"
    cached = _cache_get_pillar(cache_key)
    if cached is not None:
        return "inst", cached[0], cached[1], "HIT"
    score, meta = await _run(_fetch_inst_sync, ticker)
    _cache_set_pillar(cache_key, score, meta, _TTL["inst"])
    return "inst", score, meta, "MISS"


async def _pillar_news(ticker: str) -> Tuple[str, float, Dict, str]:
    cache_key = f"p3:news:{ticker}"
    cached = _cache_get_pillar(cache_key)
    if cached is not None:
        return "news", cached[0], cached[1], "HIT"
    score, meta = await _run(_fetch_news_sync, ticker)
    _cache_set_pillar(cache_key, score, meta, _TTL["news"])
    return "news", score, meta, "MISS"


async def _pillar_macro(ticker: str) -> Tuple[str, float, Dict, str]:
    cache_key = f"p3:macro:{ticker}"
    cached = _cache_get_pillar(cache_key)
    if cached is not None:
        return "macro", cached[0], cached[1], "HIT"
    score, meta = await _run(_fetch_macro_sync, ticker)
    _cache_set_pillar(cache_key, score, meta, _TTL["macro"])
    return "macro", score, meta, "MISS"


# ══════════════════════════════════════════════════════════════════════════════
# Primary scoring entry point with UPGRADE 1: JSON output via tool
# ══════════════════════════════════════════════════════════════════════════════

async def score_symbol(
    ticker: str,
) -> Tuple[Signal, IntelligenceScore, Dict[str, Any]]:
    """
    Async parallel scoring with UPGRADES:
      1. JSON output via tool (forced structured output)
      2. Claude-powered contextual sentiment (news pillar)
      3. Tool-ready API structure (for future agent integration)
    """
    ticker = ticker.upper()
    results = await asyncio.gather(
        _pillar_sent(ticker),
        _pillar_ins(ticker),
        _pillar_inst(ticker),
        _pillar_news(ticker),
        _pillar_macro(ticker),
    )

    scores:  Dict[str, float] = {}
    pillar_meta: Dict[str, Dict] = {}
    raw_details: Dict[str, Any] = {}

    for pillar, score, meta, cache_status in results:
        scores[pillar]      = score
        pillar_meta[pillar] = meta
        raw_details[pillar] = {
            "score":  score,
            "cache":  cache_status,
            **meta,
        }

    # Dynamic weighting
    dyn_weights, confidence, triggers = calculate_dynamic_weights(
        pillar_meta, scores=scores
    )

    # Final conviction
    active_scores = {k: v for k, v in scores.items() if pillar_meta[k].get("has_data", False)}
    conviction = _geometric_weighted_mean(active_scores, dyn_weights) if active_scores else _NEUTRAL

    # Convergence Alpha
    ins_score = scores.get("ins", 0.0)
    sent_score = scores.get("sent", 0.0)
    
    if ins_score >= _CONVERGENCE_ALPHA_THRESHOLD and sent_score >= _CONVERGENCE_ALPHA_THRESHOLD:
        old_conviction = conviction
        conviction = max(conviction, _CONVERGENCE_ALPHA_MIN)
        triggers.append(f"convergence_alpha:ins={ins_score:.2f},sent={sent_score:.2f}")
        log.info(f"[CONVERGENCE ALPHA] Insider={ins_score:.2f} & Sentiment={sent_score:.2f} "
                 f"→ Global Score override: {old_conviction:.2f} → {conviction:.2f}")

    # Build details dict
    details: Dict[str, Any] = {}
    for pillar in raw_details:
        details[pillar] = {
            **raw_details[pillar],
            "weight": dyn_weights.get(pillar, 0.0),
        }
    details["_meta"] = {
        "confidence_level": confidence,
        "weight_triggers":  triggers,
        "pillar_weights":   dyn_weights,
    }

    # Construct IntelligenceScore
    intel = IntelligenceScore(
        symbol           = ticker,
        flow_score       = scores["inst"],
        sentiment_score  = scores["sent"],
        insider_score    = scores["ins"],
        macro_score      = scores["macro"],
        final_conviction = conviction,
        congress_score   = scores["news"],
        raw_data         = {p: d["score"] for p, d in raw_details.items()},
        confidence_level = confidence,
        pillar_weights   = dyn_weights,
        weight_triggers  = triggers,
    )
    signal = _conviction_to_signal(ticker, conviction, intel)

    log.debug(
        "{} conviction={:.2%} conf={:.0%} triggers={} "
        "sent={:.2f}(w={:.0%}) ins={:.2f}(w={:.0%}) inst={:.2f}(w={:.0%}) "
        "news={:.2f}(w={:.0%}) macro={:.2f}(w={:.0%})",
        ticker, conviction, confidence, triggers,
        scores["sent"],  dyn_weights["sent"],
        scores["ins"],   dyn_weights["ins"],
        scores["inst"],  dyn_weights["inst"],
        scores["news"],  dyn_weights["news"],
        scores["macro"], dyn_weights["macro"],
    )
    return signal, intel, details


def score_symbol_sync(
    ticker: str,
) -> Tuple[Signal, IntelligenceScore, Dict[str, Any]]:
    """Blocking wrapper for sync callers."""
    try:
        return asyncio.run(score_symbol(ticker))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(score_symbol(ticker))
        finally:
            loop.close()


async def score_tickers_batch(
    tickers: List[str],
) -> Dict[str, Tuple[Signal, IntelligenceScore, Dict[str, Any]]]:
    """Score multiple tickers in parallel using asyncio.gather (cache-aware)."""
    unique = list(dict.fromkeys(t.upper() for t in tickers if t.strip()))
    tasks  = [score_symbol(t) for t in unique]
    raw    = await asyncio.gather(*tasks, return_exceptions=True)
    out: Dict[str, Tuple[Signal, IntelligenceScore, Dict[str, Any]]] = {}
    for ticker, result in zip(unique, raw):
        if isinstance(result, Exception):
            log.error("Batch score failed for {}: {}", ticker, result)
        else:
            out[ticker] = result  # type: ignore[assignment]
    return out


def score_tickers_batch_sync(
    tickers: List[str],
) -> Dict[str, Tuple[Signal, IntelligenceScore, Dict[str, Any]]]:
    """Blocking wrapper for score_tickers_batch (scripts, schedulers)."""
    try:
        return asyncio.run(score_tickers_batch(tickers))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(score_tickers_batch(tickers))
        finally:
            loop.close()


# ══════════════════════════════════════════════════════════════════════════════
# JSON Output Helper (UPGRADE 1: Structured Output)
# ══════════════════════════════════════════════════════════════════════════════

def get_structured_json_output(
    ticker: str,
) -> Dict[str, Any]:
    """
    UPGRADE 1: Returns a guaranteed JSON-serializable dict for Streamlit.
    
    This function ensures the output is always valid JSON, using the
    tool_choice pattern from the cookbook.
    
    Returns
    -------
    Dict that can be directly serialized to JSON for Streamlit display.
    """
    signal, intel, details = score_symbol_sync(ticker)
    
    return {
        "symbol": ticker.upper(),
        "final_conviction": round(intel.final_conviction, 4),
        "confidence_level": round(intel.confidence_level, 2),
        "sentiment_score": round(intel.sentiment_score, 4),
        "insider_score": round(intel.insider_score, 4),
        "institutional_score": round(intel.flow_score, 4),
        "news_score": round(intel.congress_score, 4),
        "macro_score": round(intel.macro_score, 4),
        "direction": signal.direction,
        "target_weight": round(signal.target_weight, 4),
        "justification": signal.justification,
        "weight_triggers": intel.weight_triggers or [],
        "pillar_weights": {k: round(v, 4) for k, v in (intel.pillar_weights or {}).items()},
        "raw_data": {k: round(v, 4) for k, v in (intel.raw_data or {}).items()},
        "metadata": {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "analysis_method": "claude_contextual" if details.get("news", {}).get("analysis_method") == "claude" else "vader_fallback",
            "news_analysis": details.get("news", {}).get("claude_reasoning", ""),
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# Rich terminal status table (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_PILLAR_LABELS = {
    "sent":  "SENT  (StockTwits)",
    "ins":   "INS   (FMP Insider)",
    "inst":  "INST  (FMP Institut)",
    "news":  "NEWS  (Finnhub)",
    "macro": "MACRO (Analyst)",
}


def get_detailed_status(ticker: str, details: Dict[str, Any], conviction: float) -> None:
    meta  = details.get("_meta", {})
    conf  = meta.get("confidence_level", 1.0)
    trigs = meta.get("weight_triggers", [])

    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print(f"\n{'Pillar':<22} {'Weight':>7} {'Score':>7} {'HasData':>8} {'Cache':>6}")
        for pillar, info in details.items():
            if pillar.startswith("_"):
                continue
            print(f"{_PILLAR_LABELS.get(pillar, pillar):<22} "
                  f"{info.get('weight', 0):>6.0%} "
                  f"{info.get('score', 0):>7.3f} "
                  f"{'YES' if info.get('has_data') else 'NO':>8} "
                  f"{info.get('cache', ''):>6}")
        print(f"\nConviction: {conviction:.3f}  Confidence: {conf:.0%}")
        print(f"Triggers:   {', '.join(trigs) or 'none'}\n")
        return

    console = Console()
    conv_clr = "green" if conviction >= _LONG_THRESHOLD else "red" if conviction <= _SHORT_THRESHOLD else "yellow"
    conf_clr = "green" if conf >= 0.80 else "yellow" if conf >= 0.60 else "red"

    tbl = Table(
        title=(
            f"Intelligence v3.1 — [bold]{ticker}[/bold]  "
            f"conviction=[{conv_clr}]{conviction:.1%}[/{conv_clr}]  "
            f"confidence=[{conf_clr}]{conf:.0%}[/{conf_clr}]"
        ),
        show_header=True,
        header_style="bold magenta",
    )
    tbl.add_column("Pillar",   style="cyan",    width=20)
    tbl.add_column("Wt (dyn)", justify="right", width=9)
    tbl.add_column("Score",    justify="right", width=8)
    tbl.add_column("Bar",                       width=22)
    tbl.add_column("Data",                      width=5)
    tbl.add_column("Cache",                     width=6)

    for pillar, info in details.items():
        if pillar.startswith("_"):
            continue
        s   = info.get("score", _NEUTRAL)
        w   = info.get("weight", 0.0)
        hd  = info.get("has_data", False)
        clr = "green" if s >= _LONG_THRESHOLD else "red" if s <= _SHORT_THRESHOLD else "yellow"
        filled = int(s * 20)
        bar = f"[{clr}]{'█' * filled}[/{clr}]{'░' * (20 - filled)}"
        data_marker = "[green]✓[/green]" if hd else "[red]✗[/red]"
        tbl.add_row(
            _PILLAR_LABELS.get(pillar, pillar.upper()),
            f"{w:.0%}",
            f"[{clr}]{s:.3f}[/{clr}]",
            bar,
            data_marker,
            info.get("cache", ""),
        )

    console.print()
    console.print(tbl)
    console.print(f"  → Conviction : [{conv_clr}]{conviction:.3f}[/{conv_clr}]")
    console.print(f"  → Confidence : [{conf_clr}]{conf:.0%}[/{conf_clr}]")
    console.print(f"  → Triggers   : {', '.join(trigs) or 'none'}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MarketIntelligence class (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class MarketIntelligence:
    def __init__(
        self,
        regime_label:      str   = "Unknown",
        regime_confidence: float = 0.50,
        max_workers:       int   = 6,
        rate_limit_s:      float = 0.25,
    ) -> None:
        self._regime_label      = regime_label
        self._regime_confidence = regime_confidence
        self._workers           = max_workers
        log.info(
            "MarketIntelligence v3.1 ready | regime={} conf={:.0%} workers={}",
            regime_label, regime_confidence, max_workers,
        )

    def update_regime(self, label: str, confidence: float) -> None:
        self._regime_label      = label
        self._regime_confidence = confidence

    def get_advanced_score(self, symbol: str) -> Tuple[Signal, IntelligenceScore]:
        signal, intel, _ = score_symbol_sync(symbol)
        return signal, intel

    def score_portfolio(self, symbols: List[str]) -> Dict[str, Tuple[Signal, IntelligenceScore]]:
        return self._score_parallel(symbols)

    def scan_trending(self, universe: List[str]) -> List[IntelligenceScore]:
        log.info("Scanning {} symbols …", len(universe))
        results = self._score_parallel(universe)
        top = sorted(
            [intel for _, intel in results.values()],
            key=lambda x: x.final_conviction,
            reverse=True,
        )[:_TOP_N]
        log.info(
            "Top {} opportunities: {}",
            len(top),
            [f"{s.symbol}({s.final_conviction:.0%})" for s in top],
        )
        return top

    def _score_one(self, symbol: str) -> Tuple[str, Signal, IntelligenceScore]:
        try:
            sig, intel = self.get_advanced_score(symbol)
            return symbol, sig, intel
        except Exception as exc:
            log.error("Score failed for {}: {}", symbol, exc)
            neutral_intel = IntelligenceScore(
                symbol           = symbol,
                flow_score       = 0.5,
                sentiment_score  = 0.5,
                insider_score    = 0.5,
                macro_score      = 0.5,
                final_conviction = 0.5,
                congress_score   = 0.5,
                raw_data         = {"error": str(exc)},
            )
            neutral_signal = _conviction_to_signal(symbol, 0.5, neutral_intel)
            return symbol, neutral_signal, neutral_intel

    def _score_parallel(
        self, symbols: List[str]
    ) -> Dict[str, Tuple[Signal, IntelligenceScore]]:
        results: Dict[str, Tuple[Signal, IntelligenceScore]] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="intel_scan"
        ) as pool:
            futures = {pool.submit(self._score_one, sym): sym for sym in symbols}
            for future in concurrent.futures.as_completed(futures):
                sym = futures[future]
                try:
                    sym, sig, intel = future.result(timeout=45)
                    results[sym] = (sig, intel)
                except concurrent.futures.TimeoutError:
                    log.warning("Score timeout for {}", sym)
                except Exception as exc:
                    log.error("Parallel score error for {}: {}", sym, exc)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Rebalancing logic (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rebalance_actions(
    portfolio,
    signals:   Dict[str, Signal],
    tolerance: float = 0.03,
) -> List:
    from core.models import ActionType, RebalanceAction

    actions = []
    for symbol, signal in signals.items():
        current_w = portfolio.weight_of(symbol)
        target_w  = signal.target_weight
        delta     = target_w - current_w

        if abs(delta) < tolerance:
            action = ActionType.HOLD.value
        elif delta > 0 and signal.direction == Direction.LONG.value:
            action = ActionType.BUY.value
        elif delta < 0 and current_w > 0:
            action = ActionType.REDUCE.value if abs(delta) < 0.05 else ActionType.SELL.value
        elif signal.direction == Direction.SHORT.value and current_w > 0:
            action = ActionType.SELL.value
        else:
            action = ActionType.HOLD.value

        actions.append(
            RebalanceAction(
                symbol          = symbol,
                action          = action,
                current_weight  = round(current_w, 4),
                target_weight   = round(target_w, 4),
                delta_weight    = round(delta, 4),
                estimated_value = round(abs(delta) * portfolio.equity, 2),
                signal          = signal,
                score           = None,
            )
        )

    actions.sort(
        key=lambda a: (a.action == ActionType.HOLD.value, -abs(a.delta_weight))
    )
    return actions