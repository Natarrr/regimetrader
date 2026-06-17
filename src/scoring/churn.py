# Path: src/scoring/churn.py
"""Selection churn — tenure tracking + cooldown rotation.

Pure rotation logic that breaks the "always the same tickers" trap: a name that
holds a top-N slot for ``max_tenure`` consecutive runs is forced to sit out the
next run, letting the next-best contender take its place. Tenure persists across
runs (logs/universe_state.json); every rotation is logged with a reason
(logs/universe_churn.ndjson) so the selection is auditable.

Turnover-budget intuition: cap how long a single name can monopolize capital,
concentrating fresh turnover on rotating contenders [Zhang-Wang-Cao 2021,
turnover-adjusted IR]. Gated by the UNIVERSE_CHURN flag at the call site.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Sequence, Set, Tuple


def tickers_on_cooldown(state: Dict[str, int], max_tenure: int) -> Set[str]:
    """Names whose consecutive top-N tenure has reached the rotation cap."""
    return {t for t, tenure in state.items() if tenure >= max_tenure}


def cooled_top_n(
    entries: Sequence[Dict[str, Any]],
    state: Dict[str, int],
    max_tenure: int,
    n: int,
    ticker_key: str = "ticker",
    score_key: str = "final_score",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pick the top ``n`` entries by score, rotating out over-tenured leaders.

    Returns (selected_entries, churn_events). Entries are ranked by score
    descending; any contender on cooldown that would otherwise have made the cut
    is skipped (logged ``cooldown``) and the next name is promoted (logged
    ``added``). Names retained from prior runs are not re-logged.
    """
    ranked = sorted(entries, key=lambda e: e[score_key], reverse=True)
    cooled = tickers_on_cooldown(state, max_tenure)
    today = date.today().isoformat()

    selected: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    for e in ranked:
        if len(selected) >= n:
            break
        ticker = e[ticker_key]
        if ticker in cooled:
            events.append({
                "date": today, "ticker": ticker, "action": "cooldown",
                "reason": f"held top-{n} for {state[ticker]} runs "
                          f">= max_tenure {max_tenure} — rotated out for one run",
                "prior_tenure": state[ticker],
                "score": round(float(e[score_key]), 4),
            })
            continue
        if state.get(ticker, 0) == 0:   # newly promoted into the slot
            events.append({
                "date": today, "ticker": ticker, "action": "added",
                "reason": f"promoted into top-{n} at score "
                          f"{float(e[score_key]):.4f}",
                "score": round(float(e[score_key]), 4),
            })
        selected.append(e)
    return selected, events


def update_tenure(
    prev_state: Dict[str, int],
    selected: Sequence[str],
    cooled_out: Sequence[str],
) -> Dict[str, int]:
    """Next-run tenure: increment retained leaders, reset cooled, drop the rest.

    A cooled-out name resets to 0 so it is eligible again next run (one run on
    the bench, not a permanent ban).
    """
    new: Dict[str, int] = {t: prev_state.get(t, 0) + 1 for t in selected}
    for t in cooled_out:
        new[t] = 0
    return new
