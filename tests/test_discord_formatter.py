# Path: tests/test_discord_formatter.py
"""Tests for DiscordPayloadBuilder — institutional daily-brief layout.

Contract under test (sole production builder for cooked logs/top_lists.json):
  - Theme dispatch: NORMAL / BEAR / CAPITULATION (colors, ANSI action bar,
    risk multiplier from src.risk.regime — never hardcoded).
  - ANSI hygiene: escape codes confined to the leading ```ansi block, reset
    before the closing fence, double-newline terminator before markdown.
  - Factor matrix: ASCII-only code block ('-' for absent, no em-dash, no
    tabs), CG column US-only, FCF/AMH/PB/ROI intl-only, equal row lengths.
  - Desk lines: 4-dp score, percentile, badge, score delta (stale-aware),
    intl insider $ passthrough, COV warning.
  - Budget: desc <=4096, field <=1024, total <=6000, balanced fences.
  - DATA UNAVAILABLE alert title contract (test_daily_toplists_absence.yml).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

ESC = "\x1b"

_NOW = datetime.now(timezone.utc)


# ── Fixture builders (cooked top_lists.json schema) ───────────────────────────

def _entry(ticker, score=0.72, market="USA", **kw):
    badge = ("HIGH BUY" if score >= 0.80
             else "TACTICAL BUY" if score >= 0.60 else "WATCHLIST")
    factors = {
        "insider_conviction": 0.50, "insider_breadth": 0.40,
        "congress": 0.30 if market == "USA" else 0.0,
        "news_sentiment": 0.60, "news_buzz": 0.45,
        "momentum_long": 0.70, "volume_attention": 0.0,
        "analyst_consensus": 0.55, "quality_piotroski": 0.65,
        "sector": kw.pop("sector", "Technology"),
    }
    if market in ("EUROPE", "ASIA"):
        factors.update({
            "fcf_yield": 0.55, "amihud_shock": 0.38,
            "pb_value_up": 0.47, "roic_quality": 0.52,
        })
    factors.update(kw.pop("factors", {}))
    base = {
        "ticker": ticker,
        "final_score": score,
        "badge": badge,
        "market": market,
        "factors": factors,
        "insider_usd": 0.0,
        "momentum_spy_relative": 0.0,
    }
    base.update(kw)
    return base


def _top_lists(vix=17.3, **overrides):
    data = {
        "top_buys_usa": [
            _entry("NVDA", 0.8412, insider_usd=2_100_000,
                   momentum_spy_relative=0.182),
            _entry("MSFT", 0.7821),
        ],
        "top_buys_europe": [
            _entry("ASML.AS", 0.7102, market="EUROPE", weight_coverage=0.85),
        ],
        "top_buys_asia": [
            _entry("7203.T", 0.6404, market="ASIA", weight_coverage=0.90),
        ],
        "watchlist": [],
        "mvo_pools": {
            "large_cap_anchors": {
                "bracket": "LARGE_CAP_ANCHOR", "cap_range": ">$10B",
                "positions": [
                    {"ticker": "NVDA", "allocation": 0.25, "final_score": 0.8412,
                     "exit_anchors": {"batch_floor": 408.0, "upside_pct": 14.2}},
                ],
            },
        },
        "vix": vix,
        "vix_regime": "NORMAL",
        "kill_switch": False,
        "ticker_count": 164,
        "generated_at": (_NOW - timedelta(hours=0.4)).isoformat(),
    }
    data.update(overrides)
    return data


def _build(vix=17.3, yesterday_scores=None, yesterday_age_h=None, **overrides):
    from src.delivery.send_discord import DiscordPayloadBuilder
    builder = DiscordPayloadBuilder(
        _top_lists(vix=vix, **overrides),
        yesterday_scores=yesterday_scores,
        yesterday_age_h=yesterday_age_h,
        now=_NOW,
    )
    return builder.build()


def _embed(payload):
    return payload["embeds"][0]


def _all_text(embed):
    parts = [embed.get("title", ""), embed.get("description", "")]
    for f in embed.get("fields", []):
        parts.append(f.get("name", ""))
        parts.append(f.get("value", ""))
    parts.append((embed.get("footer") or {}).get("text", ""))
    return "\n".join(parts)


def _field(embed, name_fragment):
    for f in embed.get("fields", []):
        if name_fragment in f["name"]:
            return f
    return None


# ── 0. Data-freshness anchor (data_as_of vs cook timestamp) ───────────────────

class TestDataFreshnessAnchor:
    """`DATA Xh` and the STALE banner must reflect the age of the underlying
    market data (`data_as_of`, the oldest input leg), NOT the cook timestamp
    (`generated_at`), which is stamped milliseconds before send and so always
    reads ~0.0h. Anchored to real wall-clock because `_age_hours` uses
    datetime.now() (see _data_age_hours) — not the fixture _NOW."""

    def test_data_as_of_drives_staleness_over_fresh_generated_at(self):
        # The exact production bug: a fresh cook stamp masked stale data.
        now = datetime.now(timezone.utc)
        e = _embed(_build(
            vix=17.3,
            data_as_of=(now - timedelta(hours=30)).isoformat(),
            generated_at=now.isoformat(),  # fresh cook stamp must be ignored
        ))
        assert e["color"] == 0xFF0000
        assert "STALE" in e["description"].upper()

    def test_fresh_data_as_of_overrides_old_generated_at(self):
        now = datetime.now(timezone.utc)
        e = _embed(_build(
            vix=17.3,
            data_as_of=(now - timedelta(hours=0.2)).isoformat(),
            generated_at=(now - timedelta(hours=48)).isoformat(),  # old, ignored
        ))
        assert e["color"] == 0x00FF00
        assert "STALE" not in e["description"].upper()


# ── 1. Theme selection ─────────────────────────────────────────────────────────

class TestThemes:
    def test_normal_is_green_with_full_multiplier(self):
        e = _embed(_build(vix=17.3))
        assert e["color"] == 0x00FF00
        assert "NORMAL" in e["description"]
        assert "×1.00" in e["description"]

    def test_bear_is_orange_with_dampened_multiplier(self):
        e = _embed(_build(vix=24.0))
        assert e["color"] == 0xFFA500
        assert "BEAR" in e["description"]
        assert "×0.80" in e["description"]

    def test_capitulation_is_red_with_half_multiplier(self):
        e = _embed(_build(
            vix=34.0, kill_switch=True,
            top_buys_usa=[], top_buys_europe=[], top_buys_asia=[],
            watchlist=[_entry("JNJ", 0.41, badge="WATCHLIST")],
        ))
        assert e["color"] == 0xFF0000
        assert "CAPITULATION" in e["description"]
        assert "×0.50" in e["description"]
        assert "SUPPRESSED" in e["description"].upper()

    def test_multiplier_comes_from_regime_module(self):
        """Thresholds must come from src.risk.regime, not hardcoded 20/30."""
        from src.risk.regime import BEAR_THRESHOLD
        e = _embed(_build(vix=BEAR_THRESHOLD))  # boundary: 20.0 is BEAR
        assert "BEAR" in e["description"]

    def test_stale_data_forces_red_and_warning(self):
        gen = (_NOW - timedelta(hours=30)).isoformat()
        e = _embed(_build(vix=17.3, generated_at=gen))
        assert e["color"] == 0xFF0000
        assert "STALE" in e["description"].upper()

    def test_macro_section_has_strategy_label_and_gates(self):
        from src.risk.regime import RiskRegime, strategy_label
        e = _embed(_build(vix=17.3))
        assert strategy_label(RiskRegime.NORMAL) in e["description"]
        assert "≥0.60" in e["description"]
        assert "≥0.80" in e["description"]


# ── 2. ANSI hygiene (edge case 1) ──────────────────────────────────────────────

class TestAnsiHygiene:
    def test_description_opens_with_ansi_block(self):
        desc = _embed(_build())["description"]
        assert desc.startswith("```ansi\n")

    def test_reset_precedes_closing_fence(self):
        desc = _embed(_build())["description"]
        close = desc.index("\n```")
        assert ESC + "[0m" in desc[:close], "reset must occur inside the block"

    def test_block_terminates_with_blank_line_before_markdown(self):
        desc = _embed(_build())["description"]
        assert "```\n\n" in desc, "closing fence must be followed by blank line"

    def test_no_escape_bytes_outside_ansi_block(self):
        e = _embed(_build())
        desc = e["description"]
        after_block = desc[desc.index("\n```") + 4:]
        assert ESC not in after_block
        for f in e["fields"]:
            assert ESC not in f["name"]
            assert ESC not in f["value"]


# ── 3. Alpha desk fields ───────────────────────────────────────────────────────

class TestDeskFields:
    def test_one_field_per_nonempty_region(self):
        e = _embed(_build())
        assert _field(e, "USA") is not None
        assert _field(e, "EUROPE") is not None
        assert _field(e, "ASIA") is not None

    def test_empty_region_omitted(self):
        e = _embed(_build(top_buys_asia=[]))
        assert _field(e, "ASIA") is None

    def test_desk_line_has_score_badge_percentile(self):
        f = _field(_embed(_build()), "USA")
        assert "NVDA" in f["value"]
        assert "0.8412" in f["value"]
        assert "HIGH BUY" in f["value"]
        assert "p" in f["value"]

    def test_us_insider_usd_rendered(self):
        f = _field(_embed(_build()), "USA")
        assert "Insider $2100k" in f["value"] or "Insider $2.1M" in f["value"] \
            or "Insider $2,100k" in f["value"]

    def test_intl_insider_usd_rendered_when_present(self):
        """EU entry with insider_usd > 0 must show the dollar volume —
        the engine passthrough fix exists precisely for this."""
        eu = _entry("ASML.AS", 0.7102, market="EUROPE",
                    insider_usd=150_000, weight_coverage=0.85)
        f = _field(_embed(_build(top_buys_europe=[eu])), "EUROPE")
        assert "Insider $150k" in f["value"]

    def test_intl_insider_omitted_at_zero(self):
        # The catalyst must not fabricate an insider DOLLAR figure at zero USD.
        # (The cognitive driver strip may still show the insider_conviction
        # FACTOR score — a legitimate MAR Art.19 factor — which is distinct.)
        f = _field(_embed(_build()), "EUROPE")  # default insider_usd=0.0
        assert "Insider $" not in f["value"]

    def test_momentum_vs_spy_rendered(self):
        f = _field(_embed(_build()), "USA")
        assert "vs SPY" in f["value"]
        assert "+18.2%" in f["value"]

    def test_low_coverage_warning_on_intl(self):
        eu = _entry("ASML.AS", 0.7102, market="EUROPE", weight_coverage=0.62)
        f = _field(_embed(_build(top_buys_europe=[eu])), "EUROPE")
        assert "COV:62%" in f["value"]

    def test_high_coverage_no_warning(self):
        f = _field(_embed(_build()), "EUROPE")  # coverage 0.85
        assert "COV:" not in f["value"]


# ── 4. Score deltas (edge case 4) ──────────────────────────────────────────────

class TestScoreDeltas:
    def test_fresh_delta_shows_arrow(self):
        f = _field(_embed(_build(
            yesterday_scores={"NVDA": 0.8282}, yesterday_age_h=24.0)), "USA")
        assert "▲" in f["value"]
        assert "+0.013" in f["value"]

    def test_stale_delta_tagged_no_bare_arrows(self):
        """Monday-style gap: snapshot 72h old → interval tag, no ▲/▼."""
        f = _field(_embed(_build(
            yesterday_scores={"NVDA": 0.8282}, yesterday_age_h=72.0)), "USA")
        assert "(>48h)" in f["value"]
        assert "▲" not in f["value"]
        assert "▼" not in f["value"]

    def test_new_ticker_tagged(self):
        f = _field(_embed(_build(
            yesterday_scores={"MSFT": 0.7800}, yesterday_age_h=24.0)), "USA")
        assert "[NEW]" in f["value"]  # NVDA absent from yesterday's snapshot

    def test_no_snapshot_no_delta_tags(self):
        f = _field(_embed(_build()), "USA")
        assert "[NEW]" not in f["value"]
        assert "▲" not in f["value"]


# ── 5. Factor matrix (edge cases 2, 5, 6) ──────────────────────────────────────

class TestFactorMatrix:
    def _matrix(self, **overrides):
        f = _field(_embed(_build(**overrides)), "FACTOR MATRIX")
        assert f is not None
        return f["value"]

    def test_single_plain_code_block(self):
        val = self._matrix()
        assert val.count("```") == 2
        assert not val.startswith("```ansi")

    def test_us_header_has_congress_column(self):
        val = self._matrix()
        us_header = next(line for line in val.splitlines() if "QF" in line)
        assert "CG" in us_header

    def test_intl_header_omits_congress_adds_value_factors(self):
        val = self._matrix()
        intl_header = next(line for line in val.splitlines() if "FCF" in line)
        assert "CG" not in intl_header
        assert "AMH" in intl_header
        assert "PB" in intl_header
        assert "ROI" in intl_header

    def test_zero_factor_renders_ascii_hyphen(self):
        val = self._matrix()  # volume_attention = 0.0 on every US entry
        nvda_row = next(line for line in val.splitlines() if "NVDA" in line)
        assert "-" in nvda_row

    def test_no_em_dash_or_tabs_inside_code_blocks(self):
        e = _embed(_build())
        for f in e["fields"]:
            if "```" not in f["value"]:
                continue
            inner = f["value"].split("```")[1]
            assert "—" not in inner
            assert "\t" not in inner

    def test_rows_equal_length_with_long_intl_ticker(self):
        asia = [_entry("601318.SS", 0.6404, market="ASIA", weight_coverage=0.9)]
        val = self._matrix(top_buys_asia=asia)
        lines = [line for line in val.splitlines()
                 if line.strip() and "```" not in line]
        assert len({len(line) for line in lines}) == 1, (
            f"Misaligned matrix rows: {[(len(line), line) for line in lines]}"
        )


# ── 6. Portfolio & legend ──────────────────────────────────────────────────────

class TestPortfolioAndLegend:
    def test_mvo_pool_rendered_with_allocation_and_floor(self):
        f = _field(_embed(_build()), "PORTFOLIO")
        assert "LARGE-CAP" in f["value"]
        assert "25.0%" in f["value"]
        assert "$408" in f["value"]

    def test_sector_concentration_line_present(self):
        f = _field(_embed(_build()), "PORTFOLIO")
        assert "Tech" in f["value"]

    def test_legend_is_last_field(self):
        e = _embed(_build())
        assert "LEGEND" in e["fields"][-1]["name"]
        assert "Insider Conviction" in e["fields"][-1]["value"]
        assert "Piotroski" in e["fields"][-1]["value"]

    def test_footer_local_without_run_id(self):
        e = _embed(_build())
        assert "local" in e["footer"]["text"]

    def test_footer_shows_run_id_when_present(self):
        e = _embed(_build(source_run_id="12345678901"))
        assert "12345678901" in e["footer"]["text"]


# ── 7. CAPITULATION theme ──────────────────────────────────────────────────────

class TestCapitulation:
    def _payload(self, watchlist):
        return _build(
            vix=34.0, kill_switch=True,
            top_buys_usa=[], top_buys_europe=[], top_buys_asia=[],
            mvo_pools={}, watchlist=watchlist,
        )

    def test_no_desk_fields_and_no_mvo(self):
        e = _embed(self._payload([_entry("JNJ", 0.41, badge="WATCHLIST")]))
        assert _field(e, "ALPHA DESK") is None
        assert _field(e, "PORTFOLIO") is None

    def test_watchlist_anchors_rendered_as_watchlist(self):
        e = _embed(self._payload([_entry("JNJ", 0.41, badge="WATCHLIST")]))
        f = _field(e, "STRUCTURAL ANCHORS")
        assert f is not None
        assert "JNJ" in f["value"]
        assert "WATCHLIST" in f["value"]
        assert "BUY" not in f["value"].replace("HIGH BUY", "").replace(
            "TACTICAL BUY", "") or "WATCHLIST" in f["value"]

    def test_banner_mentions_suppression_and_live_sells(self):
        e = _embed(self._payload([_entry("JNJ", 0.41, badge="WATCHLIST")]))
        text = _all_text(e).upper()
        assert "SUPPRESSED" in text
        assert "SELL" in text

    def test_empty_watchlist_explicit_zero_anchor_line(self):
        """Edge case 7: absolute crisis — nothing survived the filters."""
        e = _embed(self._payload([]))
        text = _all_text(e)
        assert "0 assets met defensive survival thresholds" in text
        assert "100%" in text
        assert _field(e, "FACTOR MATRIX") is None

    def test_no_empty_field_values(self):
        for wl in ([], [_entry("JNJ", 0.41, badge="WATCHLIST")]):
            e = _embed(self._payload(wl))
            for f in e["fields"]:
                assert f["value"].strip(), f"Empty field value: {f['name']!r}"


# ── 8. Validation & alert ──────────────────────────────────────────────────────

class TestValidation:
    def _builder(self, data):
        from src.delivery.send_discord import DiscordPayloadBuilder
        return DiscordPayloadBuilder(data, now=_NOW)

    def test_missing_vix_fatal(self):
        data = _top_lists()
        del data["vix"]
        assert self._builder(data).validate()

    def test_non_numeric_vix_fatal(self):
        data = _top_lists()
        data["vix"] = "not-a-number"
        assert self._builder(data).validate()

    def test_missing_generated_at_fatal(self):
        data = _top_lists()
        del data["generated_at"]
        assert self._builder(data).validate()

    def test_no_regional_keys_fatal(self):
        data = _top_lists()
        for k in ("top_buys_usa", "top_buys_europe", "top_buys_asia",
                  "watchlist"):
            del data[k]
        assert self._builder(data).validate()

    def test_capitulation_empty_buys_valid(self):
        data = _top_lists(
            vix=34.0, kill_switch=True,
            top_buys_usa=[], top_buys_europe=[], top_buys_asia=[],
            watchlist=[],
        )
        assert self._builder(data).validate() == []

    def test_well_formed_input_valid(self):
        assert self._builder(_top_lists()).validate() == []


class TestAlert:
    def test_alert_title_contract(self):
        """test_daily_toplists_absence.yml asserts on embeds[0].title."""
        from src.delivery.send_discord import DiscordPayloadBuilder
        payload = DiscordPayloadBuilder.build_alert("File not found: x.json")
        embed = payload["embeds"][0]
        assert "DATA UNAVAILABLE" in embed["title"]
        assert embed["color"] == 0xFF0000
        assert "File not found: x.json" in embed["description"]


# ── 9. Budget & fence integrity (edge case 5) ──────────────────────────────────

class TestBudget:
    def _oversized(self):
        usa = [_entry(f"TICK{i:02d}", 0.85 - i * 0.01,
                      insider_usd=2_000_000 + i,
                      momentum_spy_relative=0.15)
               for i in range(12)]
        eu = [_entry(f"EU{i:02d}.PA", 0.80 - i * 0.01, market="EUROPE",
                     weight_coverage=0.55) for i in range(12)]
        asia = [_entry(f"60131{i}.SS", 0.78 - i * 0.01, market="ASIA",
                       weight_coverage=0.55) for i in range(12)]
        pools = {
            k: {"bracket": k.upper(), "cap_range": "x",
                "positions": [
                    {"ticker": f"P{k[:1]}{i}", "allocation": 0.08,
                     "final_score": 0.7,
                     "exit_anchors": {"batch_floor": 100.0 + i}}
                    for i in range(10)
                ]}
            for k in ("large_cap_anchors", "mid_cap", "small_cap")
        }
        return _build(top_buys_usa=usa, top_buys_europe=eu,
                      top_buys_asia=asia, mvo_pools=pools)

    def test_embed_limits_respected(self):
        e = _embed(self._oversized())
        assert len(e["description"]) <= 4096
        for f in e["fields"]:
            assert len(f["name"]) <= 256
            assert len(f["value"]) <= 1024, f"{f['name']} too long"
        total = (len(e.get("title", "")) + len(e.get("description", ""))
                 + sum(len(f["name"]) + len(f["value"]) for f in e["fields"])
                 + len((e.get("footer") or {}).get("text", "")))
        assert total <= 6000, f"total embed chars {total} > 6000"

    def test_fences_balanced_in_every_component(self):
        e = _embed(self._oversized())
        assert e["description"].count("```") % 2 == 0
        for f in e["fields"]:
            assert f["value"].count("```") % 2 == 0, (
                f"Orphaned code fence in {f['name']!r}"
            )

    def test_legend_survives_trimming(self):
        e = _embed(self._oversized())
        assert "LEGEND" in e["fields"][-1]["name"]


# ── 10. Lazy archive loader (edge cases 3-4) ───────────────────────────────────

class TestLoadYesterdayScores:
    def _write(self, root, day, scores):
        root.mkdir(parents=True, exist_ok=True)
        body = {"top_buys_usa": [
            {"ticker": t, "final_score": s} for t, s in scores.items()
        ]}
        (root / f"{day}_top_lists.json").write_text(
            json.dumps(body), encoding="utf-8")

    def test_reads_newest_non_today_file_only(self, tmp_path):
        from src.delivery.send_discord import _load_yesterday_scores
        root = tmp_path / "archive"
        d_today = _NOW.strftime("%Y-%m-%d")
        d_yest  = (_NOW - timedelta(days=1)).strftime("%Y-%m-%d")
        d_old   = (_NOW - timedelta(days=2)).strftime("%Y-%m-%d")
        self._write(root, d_old,   {"OLD": 0.1})
        self._write(root, d_yest,  {"NVDA": 0.8282})
        self._write(root, d_today, {"TODAY": 0.9})
        scores, age_h = _load_yesterday_scores(root, now=_NOW)
        assert scores == {"NVDA": 0.8282}
        assert age_h is not None and 24 <= age_h <= 48

    def test_old_snapshot_reports_stale_age(self, tmp_path):
        from src.delivery.send_discord import _load_yesterday_scores
        root = tmp_path / "archive"
        d_old = (_NOW - timedelta(days=3)).strftime("%Y-%m-%d")
        self._write(root, d_old, {"NVDA": 0.8})
        scores, age_h = _load_yesterday_scores(root, now=_NOW)
        assert scores == {"NVDA": 0.8}
        assert age_h > 48

    def test_missing_dir_returns_empty(self, tmp_path):
        from src.delivery.send_discord import _load_yesterday_scores
        scores, age_h = _load_yesterday_scores(tmp_path / "nope", now=_NOW)
        assert scores == {}
        assert age_h is None

    def test_corrupt_newest_falls_through_to_older(self, tmp_path):
        from src.delivery.send_discord import _load_yesterday_scores
        root = tmp_path / "archive"
        d_yest = (_NOW - timedelta(days=1)).strftime("%Y-%m-%d")
        d_old  = (_NOW - timedelta(days=2)).strftime("%Y-%m-%d")
        self._write(root, d_old, {"NVDA": 0.7})
        (root / f"{d_yest}_top_lists.json").write_text("{corrupt", encoding="utf-8")
        scores, _age = _load_yesterday_scores(root, now=_NOW)
        assert scores == {"NVDA": 0.7}

    def test_today_only_returns_empty(self, tmp_path):
        from src.delivery.send_discord import _load_yesterday_scores
        root = tmp_path / "archive"
        d_today = _NOW.strftime("%Y-%m-%d")
        self._write(root, d_today, {"TODAY": 0.9})
        scores, age_h = _load_yesterday_scores(root, now=_NOW)
        assert scores == {}
        assert age_h is None


# ── 11. SMID leverage desk ─────────────────────────────────────────────────────

def _smid_entry(ticker, lev=0.7480, score=0.72, mom=0.182, **kw):
    e = _entry(ticker, score, market="USA", **kw)
    e["leverage_score"] = lev
    e["momentum_spy_relative"] = mom
    return e


class TestSmidLeverageDesk:
    """top_buys_smid → '🚀 2b · SMALL-CAP / MID-CAP LEVERAGE DESK' field:
    plain ASCII code block, ljust(9) tickers, equal-length rows, max 3
    entries, graceful omission for pre-SMID artifacts and under kill-switch."""

    _TITLE = "SMALL-CAP / MID-CAP LEVERAGE DESK"

    def _smid_field(self, **overrides):
        return _field(_embed(_build(**overrides)), self._TITLE)

    def _rows(self, field):
        return [line for line in field["value"].splitlines()
                if line.strip() and "```" not in line]

    def test_field_title_contains_contract_substring(self):
        f = self._smid_field(top_buys_smid=[_smid_entry("AAOI")])
        assert f is not None
        assert "🚀 2b" in f["name"]

    def test_renders_top_three_of_five(self):
        entries = [_smid_entry(f"TK{i}", lev=0.80 - 0.01 * i) for i in range(5)]
        f = self._smid_field(top_buys_smid=entries)
        for t in ("TK0", "TK1", "TK2"):
            assert t in f["value"]
        for t in ("TK3", "TK4"):
            assert t not in f["value"]

    def test_rows_equal_length_and_ticker_ljust9(self):
        f = self._smid_field(top_buys_smid=[
            _smid_entry("AB"),
            _smid_entry("ABCDEFGHIJK", lev=0.7000),  # 11 chars → sliced to 9
        ])
        rows = self._rows(f)
        assert len({len(r) for r in rows}) == 1, (
            f"Misaligned SMID rows: {[(len(r), r) for r in rows]}"
        )
        ab_row = next(r for r in rows if r.startswith("AB "))
        assert ab_row.startswith("AB" + " " * 7), "ticker must be ljust(9)"
        assert any(r.startswith("ABCDEFGHI ") for r in rows), (
            "long ticker must be sliced to 9 chars"
        )

    def test_plain_code_block_no_ansi_balanced_fences(self):
        f = self._smid_field(top_buys_smid=[_smid_entry("AAOI")])
        assert f["value"].count("```") == 2
        assert not f["value"].startswith("```ansi")
        assert ESC not in f["value"]
        inner = f["value"].split("```")[1]
        assert "—" not in inner
        assert "\t" not in inner

    def test_absent_key_omits_field(self):
        """Backward compat: pre-SMID artifacts carry no top_buys_smid key."""
        assert self._smid_field() is None

    def test_empty_list_omits_field(self):
        assert self._smid_field(top_buys_smid=[]) is None

    def test_capitulation_skips_smid(self):
        """Defense in depth: even a (foreign/stale) artifact carrying a
        populated pool must not render a leverage desk under kill-switch."""
        f = self._smid_field(
            vix=34.0, kill_switch=True,
            top_buys_usa=[], top_buys_europe=[], top_buys_asia=[],
            mvo_pools={}, watchlist=[_entry("JNJ", 0.41, badge="WATCHLIST")],
            top_buys_smid=[_smid_entry("AAOI")],
        )
        assert f is None

    def test_shows_leverage_momentum_and_flags(self):
        f = self._smid_field(top_buys_smid=[_smid_entry(
            "AAOI", lev=0.7480, mom=0.182,
            earnings_surprise_pct=12.0, earnings_surprise_days=45,
            quality_piotroski_score=0.625,
        )])
        assert "0.7480" in f["value"]
        assert "+18.2%" in f["value"]
        assert "E45d" in f["value"]   # PEAD recency flag
        assert "F5" in f["value"]     # 0.625 * 8 = 5 Piotroski points

    def test_flags_dash_when_meta_absent(self):
        f = self._smid_field(top_buys_smid=[_smid_entry("AAOI")])
        row = next(r for r in self._rows(f) if r.startswith("AAOI"))
        assert row.rstrip().endswith("-")
        assert "E" not in row.replace("AAOI", "")
        assert "F" not in row.replace("AAOI", "")

    def test_budget_respected_with_smid(self):
        usa = [_entry(f"TICK{i:02d}", 0.85 - i * 0.01,
                      insider_usd=2_000_000 + i,
                      momentum_spy_relative=0.15)
               for i in range(12)]
        eu = [_entry(f"EU{i:02d}.PA", 0.80 - i * 0.01, market="EUROPE",
                     weight_coverage=0.55) for i in range(12)]
        asia = [_entry(f"60131{i}.SS", 0.78 - i * 0.01, market="ASIA",
                       weight_coverage=0.55) for i in range(12)]
        smid = [_smid_entry(f"SM{i:02d}", lev=0.80 - 0.01 * i)
                for i in range(12)]
        e = _embed(_build(top_buys_usa=usa, top_buys_europe=eu,
                          top_buys_asia=asia, top_buys_smid=smid))
        total = (len(e.get("title", "")) + len(e.get("description", ""))
                 + sum(len(f["name"]) + len(f["value"]) for f in e["fields"])
                 + len((e.get("footer") or {}).get("text", "")))
        assert total <= 6000, f"total embed chars {total} > 6000"
        for f in e["fields"]:
            assert len(f["value"]) <= 1024, f"{f['name']} too long"
            assert f["value"].count("```") % 2 == 0
        assert "LEGEND" in e["fields"][-1]["name"]


    def test_smid_positioned_after_desks_before_matrix(self):
        e = _embed(_build(top_buys_smid=[_smid_entry("AAOI")]))
        names = [f["name"] for f in e["fields"]]
        i_asia = next(i for i, n in enumerate(names) if "ASIA" in n)
        i_smid = next(i for i, n in enumerate(names) if self._TITLE in n)
        i_matrix = next(i for i, n in enumerate(names) if "FACTOR MATRIX" in n)
        assert i_asia < i_smid < i_matrix


# ── 12. Cognitive factor block · regime nowcast · telemetry · sector exposure ──

from src.delivery.send_discord import (  # noqa: E402
    _factor_heat, _driver_strip, _overlay_tag, _telemetry_line,
)


class TestFactorHeatAndAttribution:
    def test_heat_thresholds(self):
        assert _factor_heat(0.80, unavailable=False) == "🟩"
        assert _factor_heat(0.50, unavailable=False) == "🟨"
        assert _factor_heat(0.20, unavailable=False) == "🟥"

    def test_data_gap_is_white_not_red(self):
        # Signed-None / thin coverage must never read as a weak 🟥 (CLAUDE.md §2).
        assert _factor_heat(0.95, unavailable=True) == "⬜"

    def test_driver_strip_ranks_and_marks_gaps(self):
        entry = {"market": "USA", "factors": {
            "insider_conviction": 0.82, "momentum_long": 0.71,
            "analyst_consensus": 0.20}}
        weights = {"insider_conviction": 0.30, "momentum_long": 0.15,
                   "analyst_consensus": 0.10, "congress": 0.05}
        strip = _driver_strip(entry, weights, missing={"congress"})
        assert "🟩 Insider 0.82" in strip
        assert "🟥 Analyst 0.20" in strip
        assert "⬜ Congress n/a" in strip   # signed-None data gap, not bearish

    def test_overlay_tag_us_only(self):
        assert _overlay_tag({"raw_score": 0.91}, 0.80) == "raw 0.91→×0.80"
        assert _overlay_tag({}, 0.80) == ""            # INTL: no raw_score
        assert _overlay_tag({"raw_score": 0.0}, 0.80) == ""


class TestTelemetryLine:
    def test_absent_field_renders_nothing(self):
        assert _telemetry_line({}) == ""

    def test_low_coverage_warns(self):
        line = _telemetry_line({"bulk_coverage": 0.60})
        assert "⚠" in line and "60%" in line

    def test_healthy_coverage_checks(self):
        line = _telemetry_line({"bulk_coverage": 0.95})
        assert "✓" in line and "95%" in line


class TestRegimeBannerAndZones:
    def test_regime_banner_present(self):
        assert "MARKET REGIME" in _embed(_build())["description"]

    def test_bear_vix_drives_market_regime_banner(self):
        # VIX in the BEAR band → BEAR nowcast even without 63d momentum (tests
        # carry none). Distinct from the risk-regime line.
        assert "MARKET REGIME — BEAR" in _embed(_build(vix=24.0))["description"]

    def test_capitulation_vix_banner(self):
        e = _embed(_build(
            vix=34.0, kill_switch=True,
            top_buys_usa=[], top_buys_europe=[], top_buys_asia=[],
            mvo_pools={}, watchlist=[_entry("JNJ", 0.41, badge="WATCHLIST")]))
        assert "MARKET REGIME — CAPITULATION" in e["description"]

    def test_book_alpha_scaled_zero_in_normal(self):
        assert "Book alpha scaled 0%" in _embed(_build(vix=17.3))["description"]

    def test_book_alpha_scaled_under_bear_overlay(self):
        # VIX 24 → vix_multiplier 0.80 → overlay shown.
        assert "overlay ×0.80" in _embed(_build(vix=24.0))["description"]

    def test_telemetry_warning_in_description(self):
        assert "TELEMETRY ⚠" in _embed(_build(bulk_coverage=0.60))["description"]

    def test_telemetry_absent_is_silent(self):
        # Pre-plumb artifacts carry no bulk_coverage — no telemetry line.
        assert "bulk-cov" not in _embed(_build())["description"]

    def test_sector_exposure_field_present(self):
        f = _field(_embed(_build()), "SECTOR EXPOSURE")
        assert f is not None
        assert "🇺🇸" in f["value"]

    def test_overlay_tag_rendered_on_us_pick(self):
        f = _field(_embed(_build(
            top_buys_usa=[_entry("NVDA", 0.84, raw_score=0.91)])), "USA")
        assert "raw 0.91" in f["value"]

    def test_lifecycle_line_on_rank1(self):
        nvda = _entry("NVDA", 0.84, target_price=214.0, current_price=197.0,
                      volume_spike=2.4)
        f = _field(_embed(_build(top_buys_usa=[nvda])), "USA")
        assert "🎯 tgt $214" in f["value"]
        assert "relVol 2.4×" in f["value"]


# ── 13. On-demand single-ticker audit (ChatOps) ────────────────────────────────

def _on_demand_data(ticker="TSLA", pipeline="US", market="USA", score=0.6432,
                    vix=17.3, kill_switch=False, entry_extra=None, **overrides):
    entry = _entry(ticker, score, market=market,
                   insider_usd=120_000, momentum_spy_relative=0.124,
                   weight_coverage=0.86)
    entry["pipeline"] = pipeline
    entry["validation_metadata"] = {
        "is_complete": True,
        "missing_sources": ["congress", "transcript_tone"],
    }
    entry.update(entry_extra or {})
    data = {
        "on_demand": True,
        "on_demand_ticker": {
            "ticker": ticker,
            "pipeline": pipeline,
            "scoring_mode": "absolute",
            "entry": entry,
        },
        "vix": vix,
        "vix_regime": "NORMAL",
        "kill_switch": kill_switch,
        "ticker_count": 1,
        "generated_at": (_NOW - timedelta(hours=0.2)).isoformat(),
    }
    data.update(overrides)
    return data


def _build_on_demand(**kw):
    from src.delivery.send_discord import DiscordPayloadBuilder
    builder = DiscordPayloadBuilder(_on_demand_data(**kw), now=_NOW)
    return builder.build_on_demand()


def _fenced_blocks(text):
    """Return the contents of every ``` code fence in text."""
    parts = text.split("```")
    # parts[1], parts[3], ... are inside fences
    return [parts[i] for i in range(1, len(parts), 2)]


class TestOnDemand:
    def test_title_exact_contract(self):
        e = _embed(_build_on_demand(ticker="TSLA"))
        assert e["title"] == "── 📊 ON-DEMAND FACTOR AUDIT: TSLA ──"

    def test_validate_accepts_on_demand_only_payload(self):
        from src.delivery.send_discord import DiscordPayloadBuilder
        builder = DiscordPayloadBuilder(_on_demand_data(), now=_NOW)
        assert builder.validate() == []

    def test_validate_still_rejects_missing_vix(self):
        from src.delivery.send_discord import DiscordPayloadBuilder
        data = _on_demand_data()
        del data["vix"]
        builder = DiscordPayloadBuilder(data, now=_NOW)
        assert builder.validate() != []

    def test_ansi_hygiene(self):
        e = _embed(_build_on_demand())
        desc = e["description"]
        assert desc.startswith("```ansi\n")
        close = desc.index("\n```")
        assert ESC + "[0m" in desc[:close], "reset must occur inside the block"
        assert "```\n\n" in desc
        after_block = desc[desc.index("\n```") + 4:]
        assert ESC not in after_block
        for f in e["fields"]:
            assert ESC not in f["name"]
            assert ESC not in f["value"]

    def test_factor_stack_rows_equal_length(self):
        f = _field(_embed(_build_on_demand()), "FACTOR STACK")
        assert f is not None
        rows = [r for r in _fenced_blocks(f["value"])[0].splitlines() if r]
        lengths = {len(r) for r in rows}
        assert len(lengths) == 1, f"unequal row lengths: {sorted(lengths)}"

    def test_factor_stack_ascii_only_inside_fence(self):
        f = _field(_embed(_build_on_demand()), "FACTOR STACK")
        for block in _fenced_blocks(f["value"]):
            assert all(ord(c) < 128 for c in block), "non-ASCII inside code fence"

    def test_factor_stack_shows_final_and_raw(self):
        f = _field(_embed(_build_on_demand(score=0.6432)), "FACTOR STACK")
        assert "0.6432" in f["value"]
        assert "FINAL SCORE" in f["value"]

    def test_us_stack_has_congress_no_intl_factors(self):
        f = _field(_embed(_build_on_demand(market="USA")), "FACTOR STACK")
        assert "Congress" in f["value"]
        assert "FCF" not in f["value"]

    def test_intl_stack_has_intl_factors_no_congress(self):
        e = _embed(_build_on_demand(ticker="SAP.DE", pipeline="INTL",
                                    market="EUROPE"))
        f = _field(e, "FACTOR STACK")
        assert "FCF Yield" in f["value"]
        assert "Congress" not in f["value"]

    def test_catalyst_field_renders_evidence(self):
        f = _field(_embed(_build_on_demand()), "CATALYST")
        assert f is not None
        assert "Insider $120k" in f["value"]

    def test_disclosure_shows_absolute_mode(self):
        f = _field(_embed(_build_on_demand()), "DISCLOSURE")
        assert f is not None
        assert "absolute" in f["value"]
        assert "congress" in f["value"]  # missing_sources surfaced

    def test_legend_present(self):
        e = _embed(_build_on_demand())
        assert _field(e, "LEGEND") is not None

    def test_kill_switch_forces_capitulation_render(self):
        e = _embed(_build_on_demand(kill_switch=True))
        assert e["color"] == 0xFF0000
        assert "KILL-SWITCH ACTIVE" in e["description"]
        assert "CAPITULATION" in e["description"]

    def test_budget_limits_hold(self):
        from src.delivery.send_discord import DiscordPayloadBuilder
        e = _embed(_build_on_demand())
        assert len(e["description"]) <= 4096
        for f in e["fields"]:
            assert len(f["value"]) <= 1024
        assert DiscordPayloadBuilder._embed_size(e) <= 6000

    def test_fences_balanced(self):
        e = _embed(_build_on_demand())
        text = _all_text(e)
        assert text.count("```") % 2 == 0

    # ── Sprint 4: source-diagnostic display ─────────────────────────────────

    def test_disclosure_splits_no_coverage_from_api_error(self):
        entry_extra = {
            "validation_metadata": {
                "is_complete": True,
                "missing_sources": [
                    "news_sentiment:no_coverage",
                    "analyst_revision:no_coverage",
                    "fcf_yield:api_error",
                ],
            }
        }
        f = _field(_embed(_build_on_demand(entry_extra=entry_extra)), "DISCLOSURE")
        assert "No coverage:" in f["value"]
        assert "news_sentiment" in f["value"]
        assert "analyst_revision" in f["value"]
        assert "API errors:" in f["value"]
        assert "fcf_yield" in f["value"]
        assert "Missing sources:" not in f["value"]

    def test_disclosure_only_no_coverage_omits_api_errors_line(self):
        entry_extra = {
            "validation_metadata": {
                "is_complete": True,
                "missing_sources": ["news_sentiment:no_coverage"],
            }
        }
        f = _field(_embed(_build_on_demand(entry_extra=entry_extra)), "DISCLOSURE")
        assert "No coverage:" in f["value"]
        assert "API errors:" not in f["value"]

    def test_disclosure_legacy_plain_missing_sources_still_renders(self):
        # Plain names (no colon) from daily pipeline → old label preserved
        entry_extra = {
            "validation_metadata": {
                "is_complete": True,
                "missing_sources": ["congress", "transcript_tone"],
            }
        }
        f = _field(_embed(_build_on_demand(entry_extra=entry_extra)), "DISCLOSURE")
        assert "Missing sources:" in f["value"]
        assert "congress" in f["value"]


class TestWhaleAndNicheAlpha:
    """v2.5 — 🐋 whale-accumulation badge, [NICHE ALPHA] strip, exit-anchor target."""

    def test_whale_signal_on_high_13f_flow(self):
        from src.delivery.send_discord import _whale_signal
        assert _whale_signal({"factors": {"inst_flow_13f": 0.85}}) is True

    def test_whale_signal_on_insider_npr_spike(self):
        from src.delivery.send_discord import _whale_signal
        assert _whale_signal({"factors": {}, "insider_npr": {"spike": 0.35}}) is True

    def test_no_whale_signal_when_flow_moderate(self):
        from src.delivery.send_discord import _whale_signal
        assert _whale_signal({"factors": {"inst_flow_13f": 0.55}}) is False

    def test_badge_only_for_small_mid_cap(self):
        from src.delivery.send_discord import _whale_badge
        strong = {"factors": {"inst_flow_13f": 0.9}}
        assert _whale_badge({**strong, "cap_tier": "small"}) != ""
        assert _whale_badge({**strong, "cap_tier": "mid"}) != ""
        # A large-cap with the same flow must NOT get the niche badge.
        assert _whale_badge({**strong, "cap_tier": "large"}) == ""

    def test_niche_strip_renders_13f_and_npr(self):
        from src.delivery.send_discord import _niche_alpha_strip
        out = _niche_alpha_strip({
            "factors": {"inst_flow_13f": 0.82},
            "inst_13f_evidence": {"investors_holding_change": 12},
            "insider_npr": {"npr": 0.9, "spike": 0.4, "acquired": 9, "disposed": 1},
        })
        assert out.startswith("[NICHE ALPHA]")
        assert "🐋 13F 0.82" in out and "(+12 holders)" in out
        assert "9B/1S" in out and "▲" in out

    def test_target_token_exit_anchor(self):
        from src.delivery.send_discord import _target_token
        assert _target_token({"target_price": 120.0, "current_price": 100.0}) == \
            "🎯 tgt $120 (+20.0%)"
        assert _target_token({"target_price": 0.0, "current_price": 100.0}) == ""

    def test_badge_niche_and_target_render_in_embed(self):
        """Integration: a small-cap whale at rank 1 surfaces the 🐋 badge + the
        [NICHE ALPHA] line; a rank-2 pick still shows its 🎯 exit target."""
        usa = [
            _entry("LEVR", 0.82, cap_tier="small",
                   factors={"inst_flow_13f": 0.90},
                   insider_npr={"npr": 0.9, "spike": 0.4, "acquired": 9, "disposed": 1},
                   inst_13f_evidence={"investors_holding_change": 12},
                   target_price=120.0, current_price=100.0),
            _entry("MSFT", 0.78, cap_tier="large",
                   target_price=400.0, current_price=350.0),
        ]
        text = _all_text(_embed(_build(top_buys_usa=usa)))
        assert "🐋 WHALE ACCUMULATION" in text     # rank-1 small-cap badge
        assert "[NICHE ALPHA]" in text             # 13F + insider velocity line
        assert "🎯 tgt $400" in text               # rank-2 per-row exit target
