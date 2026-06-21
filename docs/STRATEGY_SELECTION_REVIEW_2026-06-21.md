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

The **one material gap is universe breadth, not logic**: the investable universe
is large-cap-dominated with a thin mid-cap sleeve and **zero small-caps**, even
though the SMID machinery (ADV floors, `cap_tier` neutralization, soft-beta
rank) is built to support a "small" tier. In its current state, **"SMID"
effectively means "Mid/Large."**

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

### F1 — No small-caps; sparse mid-caps (MATERIAL)
Root cause in [tools/build_universes.py](../tools/build_universes.py):
- screener floor `marketCapMoreThan = 2_000_000_000` ($2B) → nothing below $2B is
  ever fetched;
- classifier `cap_tier = "large" if mcap >= 10e9 else "mid"` — **two-way, no
  "small" branch** — despite the module docstring defining `small < $2B` and
  `ADV_FLOOR` carrying explicit `small` floors.

Consequence: the "small" tier is structurally unreachable; the SMID strategy's
small-cap leg is dormant. **This is a universe-scope decision, not a logic bug**
— flagged for the lead quant, not changed here (SMID scope is a locked decision;
launch posture is paper-only/conservative).

*Lever to enable small-caps (if desired):* lower the screener floor (e.g. to
$3–5B mid floor and a separate small band), and make the classifier 3-way
(`small < $2B`, `mid $2–10B`, `large ≥ $10B`). Re-validate ADV floors and the
soft-beta gate against the wider, less-liquid set before any weight.

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

1. **Decide small-cap scope** (F1). If SMID should include small-caps, apply the
   lever in F1; otherwise rename/scope expectations to "Mid/Large" to avoid
   overstating reach.
2. **Broaden the mid sleeve** to clear `min_bucket_size` per sector (F2).
3. **Run one fresh full pipeline** (US + EU + APAC) to refresh `logs/` and verify
   the live selection matches this design review (F4).
4. No changes required to scoring/region/neutralization logic — it is coherent.
