# Strategy Selection Review — 2026-06-21

**Scope:** Global coherence check of the selection strategy across cap tiers
(small / mid / large) and markets (US / EU / APAC), and of the US-specific
signal palette (congress, insider, 13F, PEAD, …).
**Method:** Config + code review on `main` @ `9138239`. The `logs/*.json`
artifacts are stale and from different dates (2026-05-08 → 06-15), so they do
**not** represent one coherent run; this review judges the *universe composition*
(static config) and the *strategy logic* (code), not a live snapshot.

---

## 1. Verdict

The **scoring logic is coherent and correctly isolated**: US / EU / ASIA are
scored as three separate `score_universe_v3()` passes with market-prefixed
neutralization buckets and region masks that bar US-structural factors from
EU/ASIA. The regional theses, factor weights (sum 1.0, 3×3 pillars), and
cross-sectional neutralization are sound.

**Small-caps come from a runtime SMID satellite, not the static CSVs.** The
`config/universe*.csv` files are only the large/mid **core anchor**; the small
($300M–$2B) + mid ($2B–$10B) sleeve is screened dynamically each run by
`src/ingestion/universe_screener._resolve_smid_satellite`, gated by
`UNIVERSE_SMID_SATELLITE`.

- **US: already live.** `run_us_pipeline` sets `UNIVERSE_SMID_SATELLITE=1`
  (default, no repo-variable override) and `run_pipeline` calls
  `resolve_universe`, so small-caps enter the US selection at runtime. The
  fetcher classifies tiers correctly 3-way (`<$2B` small / `$2–10B` mid / `≥$10B`
  large).
- **EU/APAC: was the real gap (now fixed in this change).** The INTL job seeded
  EU/Asia only from `ticker_registry.json` (large/mid lists), never called the
  satellite, and its job `env` did not set the flag. Resolved here by adding
  `UNIVERSE_SMID_SATELLITE` to the `run_intl_pipeline` env and calling a new
  `smid_satellite_tickers()` helper in the INTL fetch step (per region, flag-
  gated, non-raising → degrades to registry-only on any failure).

---

## 2. Universe composition (config/)

| Market | n | large | mid | small | sectors | taxonomy |
|---|---:|---:|---:|---:|---:|---|
| US (`universe.csv`) | 160 | 139 | 21 | **0** | 11 | GICS |
| EU (`universe_eu.csv`) | 273 | 233 | 40 | **0** | 12 | Yahoo |
| APAC (`universe_apac.csv`) | 522 | 501 | 21 | **0** | 12 | Yahoo |

- EU exchanges: LSE 92, STO 89, CPH 51, HEL 19, XETRA 15, AMS/PAR/BRU.
- APAC exchanges: KSC 100, KOE 100, NSE 100, SET 94, HKSE 84, SES 37, JPX 6, SHH 1.
- All three markets are represented and sector-diversified; **only the small-cap
  tier is absent across all three.**

---

## 3. Per-market signal palette (`src/config/factor_matrix.py`)

Each region uses a distinct 3-pillar thesis; region masks (`engine_v3.assert_region_isolation`)
guarantee US-structural factors never leak into EU/ASIA.

- **US — "Alternative Alpha" (P1 0.30 / P2 0.25 / P3 0.45):** quality_dupont .12,
  fcf_yield .10, quality_piotroski .08 | analyst_revision .08, **pead_surprise .09**,
  price_target_upside .08 | **insider_alpha .30, congress .05, inst_flow_13f .10**.
  → US provides the full palette incl. **congress** (S3 primary + FMP fallback),
  Form-4 insider composite, 13F flow, PEAD. ✔
- **EU — "Quality & Value" (0.45 / 0.35 / 0.20):** piotroski, fcf_yield, pb_value_up |
  analyst_consensus, analyst_revision, price_target_upside | inst_concentration,
  dividend_sustain, amihud_shock. (No congress/insider/13F/PEAD — by design.) ✔
- **ASIA — "Growth & Reversion" (0.35 / 0.40 / 0.25):** margin_expansion,
  roic_quality, piotroski | analyst_revision, revision_velocity, price_target_upside |
  inst_concentration, dividend_sustain, amihud_shock. ✔

`US_STRUCTURAL_ONLY = {congress, inst_flow_13f, pead_surprise, insider_alpha,
quality_dupont, transcript_tone}` — masked out of EU/ASIA pools. ✔

---

## 4. Cap-tier / SMID handling

- **Neutralization** (`src/scoring/neutralization.py`): z-score within
  `(market, sector, cap_tier)` buckets, falling back to `cap_tier`-only, then
  raw, then zero. Correct peer-group isolation. ✔
- **SMID rotation** (Phase 2): ADV gate + soft-beta rank + `cap_tier` plumbing
  present and wired. ✔ — but operating on a universe with ~0 small / few mid.

---

## 5. Findings

### F1 — Small-caps reach US but not EU/APAC (RESOLVED in this change)

The static CSVs carry no small-caps **by design** (they are the core anchor).
Small-caps are added at runtime by the SMID satellite. That path was wired and
enabled for **US** but **absent for EU/APAC**:

- US `run_pipeline` → `resolve_universe` → `_resolve_smid_satellite`, with
  `UNIVERSE_SMID_SATELLITE=1` in the `run_us_pipeline` job. (live)
- EU/APAC INTL fetch seeded from `ticker_registry.json` only, no satellite call,
  and the flag was not in the `run_intl_pipeline` job `env`. (gap)

**Fix applied here:**

1. New public helper `smid_satellite_tickers(client, region, existing)` in
   [src/ingestion/universe_screener.py](../src/ingestion/universe_screener.py) —
   flag-gated, key-guarded, never raises.
2. INTL fetch step ([daily_trading_pipeline.yml](../.github/workflows/daily_trading_pipeline.yml))
   now appends the EU + ASIA small/mid sleeve per region.
3. `UNIVERSE_SMID_SATELLITE` added to the `run_intl_pipeline` job `env`
   (default `'1'`, repo-variable overridable — parity with US).

`_resolve_smid_satellite` already screens EU/ASIA exchanges and balances
small vs mid; the INTL fetcher already classifies `cap_tier` 3-way, so the new
names are labelled and neutralised correctly. The existing ADV dollar-volume
gate (`SMID_MIN_DOLLAR_VOL`, $3M/day) guards liquidity.

### F2 — Mid-cap buckets under-fill (MINOR, follows F1)
With ~21 mid-cap US names across 11 sectors (≈2/sector), most
`(sector, cap_tier=mid)` buckets fall below `min_bucket_size` and degrade to
`cap_tier`-only neutralization — coarser cross-sectional ranking for mids.
Resolves naturally if the mid sleeve is broadened.

### F3 — Sector taxonomy differs by market (INFO)
US uses GICS ("Communication Services", "Financials", "Healthcare"); EU/APAC use
Yahoo ("Financial Services", "Consumer Cyclical", "Basic Materials"). Harmless
because neutralization is per-market and buckets are market-prefixed; relevant
only for any future cross-market analytics.

### F4 — Stale artifacts (OPERATIONAL, not a strategy flaw)
`logs/latest_scores.json` (2026-05-08), `intel_source_status.json` (05-31),
`top_lists*.json` (06-06/06-15) are from different runs. A single fresh
end-to-end run is needed to judge the live selection empirically.

---

## 6. Recommendations

1. **Small-cap scope — DONE** (F1): EU/APAC now receive the SMID satellite, at
   parity with US. Next daily run will surface EU/Asia small/mid names
   (`origin="smid_satellite"`) subject to the $3M/day ADV gate.
2. **Run one fresh full pipeline** (US + EU + APAC) to refresh `logs/` and
   confirm the new EU/APAC small-caps appear (F4); inspect
   `logs/marketintel_events`/churn for the `smid_satellite` additions.
3. **Tune the sleeve if needed**: `UNIVERSE_SMID_K` (per-region count, default
   30), `SMID_MIN_DOLLAR_VOL` (liquidity floor), `SMID_BETA_ALPHA` (leverage
   tilt) — all env-overridable; repo-variable `UNIVERSE_SMID_SATELLITE=0` kills
   it globally if a run must revert to core-only.
4. No changes required to scoring/region/neutralization logic — it is coherent.
