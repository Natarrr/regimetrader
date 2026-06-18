# Design — CMF accumulation factor → ACCUMULATION WATCH (shadow-first)

**Date:** 2026-06-18
**Status:** Approved design (pre-implementation). Implementation gated on user
review of this spec, then `writing-plans`.
**Branch context:** follows the freshness/extension gate
(`feat/alpha-brief-smid-restoration`).

## Context / motivation

The Discord brief surfaces names *after* they have moved because every scored
factor is lagging or coincident (12-1m momentum; Form-4 insider, 2-3d filed over
a 180d window; 13F ~45d stale). The freshness/extension gate (already shipped)
stops us *chasing* already-moved names. This work adds the complementary half:
surface names showing **institutional accumulation before the lagging score
catches them** — a genuinely *leading* signal — without risking live capital.

Hard constraints (CLAUDE.md): evidence-first alpha (IC-validated before any
weight), strict orthogonality, no look-ahead, `assert sum(WEIGHTS)=1` integrity.
Dual-engine reality (`[[project_dual_scoring_engines]]`): v2.2-global is LIVE,
v3.0-pillars is SHADOW and the migration target — new factor work lands in v3
shadow, never as a v2.2 "fix".

## Decisions locked (from brainstorming)

1. **Direction:** add a NEW leading factor, shadow-only — zero LIVE risk until it
   clears an IC bar. (Not steepening insider decay; not options flow — FMP has no
   options data as wired.)
2. **Signal:** Chaikin Money Flow, 21-session window (`accumulation_cmf`).
3. **Surfacing during shadow:** a separate **display-only ACCUMULATION WATCH**
   Discord list of high-CMF names not yet in the score-ranked desks.
4. **Promotion bar:** reuse the existing `ic_metrics.weight_recommendation`
   ("increase") + embargo-corrected significance — no invented thresholds.

Open items deferred to the user at spec review: the **21d CMF window** and the
**`n_effective ≥ 20`** promotion threshold.

## Component 1 — the factor `score_accumulation_cmf`

- New pure scorer in `src/scoring/` over realized OHLCV arrays.
- Per-bar Money Flow Multiplier `MFM = ((C-L)-(H-C))/(H-L)`; Money Flow Volume
  `MFV = MFM * volume`; `CMF = Σ MFV(21) / Σ volume(21)` ∈ [-1, 1]; linear map to
  [0, 1] for the score.
- **None** (not 0.0) on insufficient history (<21 bars) or any zero-range bar
  guard — data absence must never read as bearish (CLAUDE.md §2). UNSIGNED in
  spirit but absence→None so it is excluded from bucket stats, not penalized.
- **Look-ahead-safe:** realized closes/highs/lows/volumes only.
- **Data dependency (must-fix):** `fetch_price_data` and `fmp_fetcher` currently
  take `closes, volumes` from `fmp_prices_to_arrays`; CMF needs **highs/lows**
  too. Extend the array helper/fetch to surface H/L from the same FMP
  `historical-price-eod/full` payload (no new endpoint).
- Tests (TDD): accumulation tape → high score; distribution tape → low; flat/zero
  range → None; bounded [0,1]; <21 bars → None.

## Component 2 — shadow integration (no LIVE impact)

- Computed inside the existing v3 shadow path `compute_v3_raw_columns` (gated by
  `SCORING_V3_SHADOW`) as raw column `accumulation_cmf`.
- **NOT** added to `FACTOR_MATRIX_V3` weights — it never enters any score during
  shadow. `assert sum(WEIGHTS)=1` is untouched.
- Added to `research/scripts/backfill_factors.py` so `ic_engine.py` measures its
  rank-IC each run. Horizon matches the existing 21d, embargo-corrected pipeline
  exactly (no overlap inflation — `[[project_ic_overlap_embargo]]`).

## Component 3 — ACCUMULATION WATCH (display-only Discord block)

- CMF scored across the **universe** (not just top picks) to form a rankable pool
  — a lightweight pass in the pipeline loop, or a dedicated scorer over the
  universe closes already fetched.
- `cook_toplists.py` builds an `accumulation_watch` list (top-CMF names that are
  **not** already present in the score-ranked desks), mirroring how `watchlist`
  flows today.
- `send_discord.py` renders **`📈 ACCUMULATION WATCH (UNSCORED — EARLY)`**: each
  line `TICKER · CMF X.XX · sector`, labelled clearly as advisory/unscored.
  Respects the 6000-char / 1024-field budget via the existing degradation ladder;
  suppressed under CAPITULATION (consistent with buy-signal suppression).
- This is the largest new moving part (universe-wide scoring + a new list key +
  render). It is purely additive and display-only.

## Component 4 — promotion bar (advisory → weighted → LIVE)

Aligned to the existing `src/research/ic_metrics.weight_recommendation`:

- **Advisory → weighted (still v3 shadow)** when `accumulation_cmf` earns
  **"increase"** (`ic_ir > 0.5` AND `ic_positive_rate ≥ 0.60`) AND embargo-
  corrected `ic_t_stat ≥ 2` over `n_effective ≥ 20` independent 21d windows, AND
  passes `monitoring/factor_orthogonality.py` vs `volume_attention` /
  `momentum_long` (low collinearity — a leading signal must be distinct from the
  undirected volume spike and from price momentum).
- **Weighted → LIVE** only after it holds the bar in v3 weighted shadow and v3 is
  itself the live engine — a separate, later human decision with the
  `assert sum(WEIGHTS)=1` invariant re-checked.

## Data flow

```
historical-price-eod/full ──► fmp_prices_to_arrays (extend: +highs/lows)
   │
   ├─ fetch_price_data (US) ─────────┐
   └─ fmp_fetcher (INTL) ────────────┤  raw OHLCV arrays
                                      ▼
                       score_accumulation_cmf  ── raw column ──► compute_v3_raw_columns (shadow)
                                      │                                  │
                                      │                                  └─► backfill_factors ─► ic_engine ─► ic_report_v3.json
                                      ▼
                       universe CMF pool ─► cook_toplists.accumulation_watch ─► send_discord 📈 ACCUMULATION WATCH
```

## Out of scope

- Insider recency-steepening / velocity up-weight (separate track).
- Options / IV / put-call flow (no FMP data as wired).
- Any change to `final_score`, `WEIGHTS`, or the LIVE v2.2 engine.

## Testing strategy

- Unit (TDD): `score_accumulation_cmf` behaviors above; H/L array extension.
- Integration: backfill emits `accumulation_cmf`; `ic_engine --engine v3` reports
  it; cook produces `accumulation_watch` excluding already-ranked tickers;
  send_discord renders the block and holds the embed budget; CAPITULATION
  suppresses it.
- Validation (operational, not a code test): accumulate snapshots, run
  `ic_engine`, evaluate against the promotion bar before any weighting.
