# Path: tests/test_send_toplists_discord.py
"""Integration tests for send_discord.main() — CLI alert paths & exit codes.

Contract (preserved across the DiscordPayloadBuilder consolidation, asserted
live by .github/workflows/test_daily_toplists_absence.yml):
  exit 2  → DISCORD_WEBHOOK_URL unset and no --webhook
  exit 0  → briefing sent, or DATA UNAVAILABLE alert sent successfully
  exit 1  → parse/validation failure (non-dry), or all send attempts failed
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

_WEBHOOK = "http://localhost:9/test-webhook"


def _cooked(tmp_path: Path, **overrides) -> Path:
    data = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "vix":             17.0,
        "vix_regime":      "NORMAL",
        "kill_switch":     False,
        "ticker_count":    1,
        "top_buys_usa":    [{
            "ticker": "MSFT", "final_score": 0.72, "badge": "TACTICAL BUY",
            "market": "USA",
            "factors": {"insider_conviction": 0.6, "momentum_long": 0.7,
                        "sector": "Technology"},
        }],
        "top_buys_europe": [],
        "top_buys_asia":   [],
        "watchlist":       [],
        "mvo_pools":       {},
    }
    data.update(overrides)
    path = tmp_path / "top_lists.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _extract_payload(out: str) -> dict:
    """Pull the JSON payload out of mixed log + JSON stdout."""
    idx = out.find('{\n  "embeds"')
    assert idx >= 0, f"No payload JSON in output: {out[:400]!r}"
    return json.loads(out[idx:])


def _run_main(args, monkeypatch, webhook=_WEBHOOK):
    from src.delivery import send_discord
    if webhook is None:
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    else:
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", webhook)
    return send_discord.main(args)


class TestExitCodes:
    def test_no_webhook_exits_2(self, tmp_path, monkeypatch):
        rc = _run_main(
            ["--input", str(tmp_path / "x.json"), "--log-dir", str(tmp_path)],
            monkeypatch, webhook=None)
        assert rc == 2

    def test_dry_run_works_without_webhook(self, tmp_path, monkeypatch, capfd):
        """README contract: --dry-run previews with no webhook configured."""
        path = _cooked(tmp_path)
        monkeypatch.setattr("src.delivery.audit_payload.audit", lambda p: None)
        rc = _run_main(
            ["--input", str(path), "--log-dir", str(tmp_path), "--dry-run"],
            monkeypatch, webhook=None)
        assert rc == 0
        payload = _extract_payload(capfd.readouterr().out)
        assert "REGIME TRADER" in payload["embeds"][0]["title"]

    def test_missing_file_alert_sent_exits_0(self, tmp_path, monkeypatch):
        """Mirrors test_daily_toplists_absence.yml: absent artifact → alert → 0."""
        sent = {}

        def _fake_send(webhook, payload, **kw):
            sent["payload"] = payload
            return True

        monkeypatch.setattr(
            "src.delivery.send_discord.send_to_discord", _fake_send)
        rc = _run_main(
            ["--input", str(tmp_path / "nope.json"), "--log-dir", str(tmp_path)],
            monkeypatch)
        assert rc == 0
        assert "DATA UNAVAILABLE" in sent["payload"]["embeds"][0]["title"]

    def test_missing_file_send_failure_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.delivery.send_discord.send_to_discord",
            lambda *a, **kw: False)
        rc = _run_main(
            ["--input", str(tmp_path / "nope.json"), "--log-dir", str(tmp_path)],
            monkeypatch)
        assert rc == 1

    def test_corrupt_json_non_dry_exits_1(self, tmp_path, monkeypatch):
        bad = tmp_path / "top_lists.json"
        bad.write_text("{corrupt", encoding="utf-8")
        monkeypatch.setattr(
            "src.delivery.send_discord.send_to_discord",
            lambda *a, **kw: True)
        rc = _run_main(
            ["--input", str(bad), "--log-dir", str(tmp_path)], monkeypatch)
        assert rc == 1


class TestDryRunPayloads:
    def test_missing_file_prints_alert(self, tmp_path, monkeypatch, capfd):
        rc = _run_main(
            ["--input", str(tmp_path / "nope.json"),
             "--log-dir", str(tmp_path), "--dry-run"],
            monkeypatch)
        assert rc == 0
        payload = _extract_payload(capfd.readouterr().out)
        assert "DATA UNAVAILABLE" in payload["embeds"][0]["title"]

    def test_corrupt_json_prints_alert(self, tmp_path, monkeypatch, capfd):
        bad = tmp_path / "top_lists.json"
        bad.write_text("{corrupt", encoding="utf-8")
        rc = _run_main(
            ["--input", str(bad), "--log-dir", str(tmp_path), "--dry-run"],
            monkeypatch)
        assert rc == 0
        payload = _extract_payload(capfd.readouterr().out)
        assert "DATA UNAVAILABLE" in payload["embeds"][0]["title"]

    def test_validation_failure_prints_alert(self, tmp_path, monkeypatch, capfd):
        path = _cooked(tmp_path)
        blob = json.loads(path.read_text(encoding="utf-8"))
        del blob["vix"]
        path.write_text(json.dumps(blob), encoding="utf-8")
        monkeypatch.setattr("src.delivery.audit_payload.audit", lambda p: None)
        rc = _run_main(
            ["--input", str(path), "--log-dir", str(tmp_path), "--dry-run"],
            monkeypatch)
        assert rc == 0
        payload = _extract_payload(capfd.readouterr().out)
        assert "DATA UNAVAILABLE" in payload["embeds"][0]["title"]
        assert "vix" in payload["embeds"][0]["description"]

    def test_valid_input_prints_briefing(self, tmp_path, monkeypatch, capfd):
        path = _cooked(tmp_path)
        monkeypatch.setattr("src.delivery.audit_payload.audit", lambda p: None)
        rc = _run_main(
            ["--input", str(path), "--log-dir", str(tmp_path), "--dry-run"],
            monkeypatch)
        assert rc == 0
        payload = _extract_payload(capfd.readouterr().out)
        embed = payload["embeds"][0]
        assert "REGIME TRADER" in embed["title"]
        assert embed["color"] == 0x00FF00
        assert any("LEGEND" in f["name"] for f in embed["fields"])


def test_module_loads_standalone_by_path():
    """send_discord.py must stay importable by file path (CI smoke pattern)."""
    spec = importlib.util.spec_from_file_location(
        "send_discord_standalone",
        Path("src/delivery/send_discord.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "DiscordPayloadBuilder")
    assert hasattr(mod, "main")


def test_audit_gate_failure_sends_alert(tmp_path, monkeypatch):
    """PipelineAuditError from the pre-flight audit → alert + exit 1."""
    from src.delivery.audit_payload import PipelineAuditError

    path = _cooked(tmp_path)
    sent = {}

    def _fake_send(webhook, payload, **kw):
        sent["payload"] = payload
        return True

    def _failing_audit(p):
        raise PipelineAuditError("score out of range")

    monkeypatch.setattr("src.delivery.audit_payload.audit", _failing_audit)
    monkeypatch.setattr("src.delivery.send_discord.send_to_discord", _fake_send)
    rc = _run_main(["--input", str(path), "--log-dir", str(tmp_path)],
                   monkeypatch)
    assert rc == 1
    assert "AUDIT GATE FAILED" in sent["payload"]["embeds"][0]["description"]
