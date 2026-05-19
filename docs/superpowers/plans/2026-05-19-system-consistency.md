# System-Wide Consistency Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four inter-system discrepancies: stale hybrid_pipeline scoring model, missing archive accumulation, stale local Stock Picker data, and a dead `_load_discovery` loader in Streamlit.

**Architecture:** Four independent changes to three files and two workflows. Each change is self-contained and can be verified independently. No new modules, no new dependencies — all changes use existing imports and libraries.

**Tech Stack:** Python 3.11, GitHub Actions YAML, Streamlit, `requests` (stdlib), `zipfile` (stdlib), `dawidd6/action-download-artifact@v8`

---

## File Map

| File | Change |
|---|---|
| `.github/workflows/hybrid_pipeline.yml` | Rewrite `quant` job + `detect regime` step |
| `.github/workflows/edgar_3x.yml` | Add `archive-snapshot` job |
| `.gitignore` | Add `!logs/archive/` exemption |
| `pages/6_Stock_Picker.py` | Add `_sync_from_github()` + Sync button |
| `regime_trader/ui/streamlit_app.py` | Remove `_load_discovery` + its `.clear()` call |

---

## Task 1: Remove dead `_load_discovery` from streamlit_app

**Context:** `_load_discovery()` (lines 205-223) is a `@st.cache_data`-wrapped function that calls `get_top_alpha_picks_sync`. It is never called to render anything — the Market Intel tab reads from `_load_market_state()` exclusively. Its only reference is a `.clear()` call in the sidebar cache button.

**Important:** `test_streamlit_app_smoke.py` does NOT reference `_load_discovery` — it tests `discovery_scanner` directly and the app import. Removing the function will not break any tests.

**Files:**
- Modify: `regime_trader/ui/streamlit_app.py:205-223` (function body)
- Modify: `regime_trader/ui/streamlit_app.py:710` (cache clear call)

- [ ] **Step 1: Verify `_load_discovery` is not referenced in tests**

```
grep -r "_load_discovery" tests/ regime_trader/
```

Expected: only matches in `regime_trader/ui/streamlit_app.py` itself (lines 206 and 710). No test file should reference it.

- [ ] **Step 2: Remove `_load_discovery` function**

In `regime_trader/ui/streamlit_app.py`, delete lines 205-223 in their entirety:

```python
# DELETE this entire block (lines 205-223):
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_discovery(limit: int = 5) -> Dict[str, Any]:
    """Return discovery picks; uses module-level TTL cache internally.

    Returns _safe_payload() on any exception so the UI never crashes on
    a scanner failure.

    Args:
        limit: Number of top picks to request.

    Returns:
        Discovery payload dict with at minimum a 'results' list.
    """
    try:
        from regime_trader.scanners.discovery_scanner import get_top_alpha_picks_sync
        return get_top_alpha_picks_sync(limit=limit)
    except Exception as exc:
        log.warning("discovery load failed: %s", exc)
        return _safe_payload()
```

The line immediately before this block is line 203 (`return None`) and immediately after is a blank line before `@st.cache_data(ttl=3600, show_spinner=False)` for `_load_commodity_prices`. Remove the function and leave one blank line between `_load_market_state` and `_load_commodity_prices`.

- [ ] **Step 3: Remove `_load_discovery.clear()` from the cache button**

Find this block in `regime_trader/ui/streamlit_app.py` (around line 708):

```python
            if st.button("Clear engine state cache", key="clear_disc"):
                _load_market_state.clear()
                _load_discovery.clear()
                st.success("Engine state + discovery cache cleared.")
```

Change it to:

```python
            if st.button("Clear engine state cache", key="clear_disc"):
                _load_market_state.clear()
                st.success("Engine state cache cleared.")
```

- [ ] **Step 4: Run smoke tests**

```
pytest tests/test_streamlit_app_smoke.py -v
```

Expected: all tests pass. Specifically `test_streamlit_app_imports` must pass (the module must import without error).

- [ ] **Step 5: Commit**

```
git add regime_trader/ui/streamlit_app.py
git commit -m "refactor(ui): remove dead _load_discovery loader

_load_discovery() called get_top_alpha_picks_sync but was never used
to render anything. The Market Intel tab reads exclusively from
_load_market_state() -> data/market_state.json (engine_worker output).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Add `.gitignore` exemption for `logs/archive/`

**Context:** `logs/` is fully gitignored (line 71 of `.gitignore`). The `edgar_3x` archive job (Task 3) will commit files to `logs/archive/`. Git must track those files. The exemption `!logs/archive/` overrides the parent `logs/` ignore rule.

**Files:**
- Modify: `.gitignore:71`

- [ ] **Step 1: Add the exemption**

In `.gitignore`, find the line:

```
logs/
```

Change the surrounding block to:

```
# ── Logs & runtime artifacts ──────────────────────────────────────────────────
logs/
!logs/archive/
*.log
*.log.*
```

The `!logs/archive/` line must come **after** `logs/` — Git processes `.gitignore` rules in order; a negation after the ignore restores tracking.

- [ ] **Step 2: Verify the exemption works**

```
git check-ignore -v logs/archive/test.json
```

Expected: no output (file is NOT ignored). If it still shows the `logs/` rule, the order is wrong — re-check step 1.

Also verify regular logs are still ignored:

```
git check-ignore -v logs/main.log
```

Expected: `.gitignore:71:logs/  logs/main.log` (still ignored).

- [ ] **Step 3: Create the archive directory and add a .gitkeep**

```
mkdir -p logs/archive
touch logs/archive/.gitkeep
git add logs/archive/.gitkeep
git commit -m "chore: exempt logs/archive/ from gitignore, add .gitkeep

archive-snapshot job in edgar_3x.yml will commit daily top_lists.json
snapshots here for the weekly_backtest workflow to consume.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Add `archive-snapshot` job to `edgar_3x.yml`

**Context:** `weekly_backtest.yml` reads `logs/archive/*.json` but nothing populates it. This job runs after `fetch-and-rank`, downloads the artifact it just uploaded, and commits it as `logs/archive/YYYY-MM-DD_top_lists.json`. Only the first run of each UTC calendar day archives — subsequent same-day runs skip silently.

**Files:**
- Modify: `.github/workflows/edgar_3x.yml` — append new job after the existing job

- [ ] **Step 1: Add the `archive-snapshot` job**

Open `.github/workflows/edgar_3x.yml`. At the very end of the file (after the last line of the `fetch-and-rank` job, which ends with `logs/*.log`), append:

```yaml

  # ── Archive daily snapshot for weekly_backtest ───────────────────────────────
  #
  # Commits logs/archive/YYYY-MM-DD_top_lists.json once per UTC calendar day.
  # Only the first successful edgar_3x run of each day archives — subsequent
  # same-day runs skip silently (git ls-files check).
  # weekly_backtest.yml reads these committed files to compute backtest metrics.
  archive-snapshot:
    name: Archive Daily Snapshot
    needs: fetch-and-rank
    runs-on: ubuntu-latest
    if: success()

    permissions:
      contents: write

    steps:

      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 1
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Download top-lists artifact
        uses: dawidd6/action-download-artifact@v8
        with:
          workflow: edgar_3x.yml
          name: top-lists
          path: logs/
          workflow_conclusion: success
          if_no_artifact_found: warn

      - name: Archive snapshot (first run of day only)
        run: |
          TODAY=$(date -u '+%Y-%m-%d')
          DEST="logs/archive/${TODAY}_top_lists.json"

          if [ ! -f logs/top_lists.json ]; then
            echo "top_lists.json not found — skipping archive"
            exit 0
          fi

          # Skip if already archived today (another run beat us to it)
          if git ls-files --error-unmatch "$DEST" 2>/dev/null; then
            echo "Archive already exists for $TODAY — skipping"
            exit 0
          fi

          mkdir -p logs/archive
          cp logs/top_lists.json "$DEST"

          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add "$DEST"
          git commit -m "chore(archive): snapshot $TODAY"
          git push
```

- [ ] **Step 2: Verify YAML syntax**

```
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/edgar_3x.yml'))"
```

Expected: no output (no exception). If you get a YAML parse error, fix the indentation — YAML is whitespace-sensitive.

- [ ] **Step 3: Commit**

```
git add .github/workflows/edgar_3x.yml
git commit -m "feat(ci): add archive-snapshot job to edgar_3x

Commits logs/archive/YYYY-MM-DD_top_lists.json once per UTC day so
weekly_backtest.yml has historical snapshots to backtest against.
Only the first successful run of each day archives; subsequent runs
skip silently.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Fix `hybrid_pipeline.yml` — consume edgar_3x artifact

**Context:** The `quant` job currently calls `discovery_scanner.get_top_alpha_picks_sync()` (3-weight model) and references `secrets.SEC_USER_AGENT` (wrong name — code reads `EDGAR_USER_AGENT`). It must instead download the `top-lists` artifact from the latest `edgar_3x` run and reshape its `top_buys` entries into the `shortlist.json` schema the downstream `claude` job expects.

**Schema the `claude` job reads from `data/pipeline/shortlist.json`:**
```json
{
  "shortlist": ["AAPL", "MSFT"],
  "candidates": [
    {
      "symbol": "AAPL",
      "smart_money_score": 0.636,
      "insider_score": 0.45,
      "inst_score": 0.60,
      "momentum_score": 0.55,
      "edgar_score": 0.70
    }
  ]
}
```

**Mapping from `top_lists.json` entry → candidate dict:**
- `symbol` ← `entry["ticker"]`
- `smart_money_score` ← `entry["final_score"]`  (already 0–1)
- `insider_score` ← `entry["factors"]["insider"]`
- `inst_score` ← `entry["factors"]["congress"]`  (institutional proxy)
- `momentum_score` ← `entry["factors"]["momentum"]`
- `edgar_score` ← `entry["factors"]["edgar"]`

**Regime derivation from `top_lists.json`:**
- `kill_switch == true` → `"Panic/Crash"`
- `vix >= 25` → `"Bear"`
- `vix < 25` or `vix` is null → `"Normal"`

**Files:**
- Modify: `.github/workflows/hybrid_pipeline.yml:68-167` (entire `quant` job)

- [ ] **Step 1: Replace the entire `quant` job**

In `.github/workflows/hybrid_pipeline.yml`, replace the `quant:` job block (everything from `  quant:` through the closing line of the `Upload shortlist artifact` step) with:

```yaml
  # ── 1. Quant scoring — consume edgar_3x top-lists artifact ─────────────────
  #
  # Downloads the top-lists artifact from the latest successful edgar_3x run
  # (same pattern as daily_toplists_discord.yml) and reshapes top_buys into
  # the shortlist.json schema the claude job expects.
  #
  # Dependency timing: edgar_3x 08:00 UTC run completes ~08:40.
  # hybrid_pipeline fires at 12:30 UTC — safe margin of ~4 hours.
  # dawidd6 always fetches the most recent successful artifact.
  quant:
    name: Quant Scoring (from edgar_3x artifact)
    runs-on: ubuntu-latest
    timeout-minutes: 10
    outputs:
      shortlist_artifact: ${{ steps.export.outputs.artifact_name }}
      regime: ${{ steps.regime.outputs.regime }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Download top-lists artifact from edgar_3x
        uses: dawidd6/action-download-artifact@v8
        with:
          workflow: edgar_3x.yml
          name: top-lists
          path: logs/
          workflow_conclusion: success
          if_no_artifact_found: warn

      - name: Reshape top_lists.json → shortlist.json
        run: |
          python3 - <<'EOF'
          import json, os, sys

          src = "logs/top_lists.json"
          if not os.path.exists(src):
              print("WARNING: top_lists.json not found — producing empty shortlist")
              output = {"shortlist": [], "candidates": []}
          else:
              with open(src) as f:
                  tl = json.load(f)

              candidates = []
              for e in tl.get("top_buys", []):
                  factors = e.get("factors", {})
                  candidates.append({
                      "symbol":            e["ticker"],
                      "smart_money_score": float(e.get("final_score", 0.0)),
                      "insider_score":     float(factors.get("insider", 0.0)),
                      "inst_score":        float(factors.get("congress", 0.0)),
                      "momentum_score":    float(factors.get("momentum", 0.0)),
                      "edgar_score":       float(factors.get("edgar", 0.0)),
                  })

              shortlist = [c["symbol"] for c in candidates]
              output = {"shortlist": shortlist, "candidates": candidates}
              print(f"Shortlist ({len(shortlist)} symbols): {shortlist}")

          os.makedirs("data/pipeline", exist_ok=True)
          with open("data/pipeline/shortlist.json", "w") as f:
              json.dump(output, f, indent=2)
          EOF

      - name: Detect regime from top_lists.json
        id: regime
        run: |
          python3 - <<'EOF'
          import json, os

          src = "logs/top_lists.json"
          regime = "Unknown"
          if os.path.exists(src):
              with open(src) as f:
                  tl = json.load(f)
              if tl.get("kill_switch"):
                  regime = "Panic/Crash"
              else:
                  vix = tl.get("vix")
                  if vix is not None:
                      regime = "Bear" if float(vix) >= 25 else "Normal"
                  else:
                      regime = "Normal"

          os.makedirs("data/pipeline", exist_ok=True)
          with open("data/pipeline/regime.txt", "w") as f:
              f.write(regime)
          print(f"Regime: {regime}")
          EOF
          REGIME=$(cat data/pipeline/regime.txt)
          echo "regime=$REGIME" >> $GITHUB_OUTPUT
          echo "Current regime: $REGIME"

      - name: Export artifact name
        id: export
        run: echo "artifact_name=quant-shortlist-${{ github.run_id }}" >> $GITHUB_OUTPUT

      - name: Upload shortlist artifact
        uses: actions/upload-artifact@v4
        with:
          name: quant-shortlist-${{ github.run_id }}
          path: data/pipeline/
          retention-days: 7
```

- [ ] **Step 2: Remove unused `Install quant deps` step**

The old `quant` job installed `hmmlearn>=0.3.0 scikit-learn>=1.4.0`. The new job runs pure Python stdlib + json — no pip install needed. Confirm the replacement above does NOT have an `Install quant deps` step. It should not — the reshaping script only uses `json`, `os`, `sys`.

- [ ] **Step 3: Verify YAML syntax**

```
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/hybrid_pipeline.yml'))"
```

Expected: no output (no exception).

- [ ] **Step 4: Verify the `claude` job still references the correct artifact name**

In the unchanged `claude` job, find the `Download shortlist artifact` step:

```yaml
      - name: Download shortlist artifact
        uses: actions/download-artifact@v4
        with:
          name: ${{ needs.quant.outputs.shortlist_artifact }}
          path: data/pipeline/
```

This still works because `steps.export.outputs.artifact_name` is still set in the new `quant` job. No change needed.

- [ ] **Step 5: Commit**

```
git add .github/workflows/hybrid_pipeline.yml
git commit -m "fix(ci): rewire hybrid_pipeline quant job to consume edgar_3x artifact

Previously called discovery_scanner (3-weight model) independently and
referenced secrets.SEC_USER_AGENT (wrong name — code reads EDGAR_USER_AGENT).

Now downloads the top-lists artifact from the latest successful edgar_3x
run and reshapes top_buys into shortlist.json for the Claude analysis job.
This ensures Claude enriches the same tickers Discord reports.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Add "Sync from GitHub" button to Stock Picker

**Context:** `pages/6_Stock_Picker.py` reads `logs/top_lists.json` from disk. This file is not auto-synced from GitHub Actions. The new button downloads the latest `top-lists` artifact and writes the file atomically only if the artifact is newer than the local copy.

**Where the button goes:** Inside the existing `with st.expander(...)` block in `render()`, after the last-run status display and before the "▶ Run edgar_3x now" button. It is gated by `if _gh_pat:` (already the case for the trigger button).

**New function signature:**
```python
def _sync_from_github(pat: str) -> tuple[bool, str]:
    """Download the latest top-lists artifact and write to logs/top_lists.json.
    Returns (success, message).
    """
```

**Files:**
- Modify: `pages/6_Stock_Picker.py` — add `_sync_from_github()` function + button in `render()`

- [ ] **Step 1: Add `_sync_from_github()` function**

In `pages/6_Stock_Picker.py`, add the following function after `_trigger_pipeline()` (after line 59, before the `@st.cache_data` decorator for `_fetch_last_run_status`):

```python
def _sync_from_github(pat: str) -> tuple[bool, str]:
    """Download latest top-lists artifact → logs/top_lists.json.

    Compares generated_at timestamps: only writes if the artifact is newer
    than the local file. Returns (success, message).
    """
    import io
    import zipfile
    try:
        import requests
    except ImportError:
        return False, "requests not installed"

    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # 1. Find the latest non-expired top-lists artifact
    list_url = (
        f"https://api.github.com/repos/{_GH_OWNER}/{_GH_REPO}"
        "/actions/artifacts?name=top-lists&per_page=5"
    )
    try:
        resp = requests.get(list_url, headers=headers, timeout=10)
    except Exception as exc:
        return False, f"Request failed: {exc}"

    if resp.status_code != 200:
        return False, f"GitHub API returned {resp.status_code}"

    artifacts = [a for a in resp.json().get("artifacts", []) if not a.get("expired")]
    if not artifacts:
        return False, "No non-expired top-lists artifact found"

    artifact_id = artifacts[0]["id"]
    created_at  = artifacts[0].get("created_at", "")

    # 2. Download the zip (GitHub redirects to a signed S3 URL)
    zip_url = (
        f"https://api.github.com/repos/{_GH_OWNER}/{_GH_REPO}"
        f"/actions/artifacts/{artifact_id}/zip"
    )
    try:
        zip_resp = requests.get(zip_url, headers=headers, timeout=30, allow_redirects=True)
    except Exception as exc:
        return False, f"Zip download failed: {exc}"

    if zip_resp.status_code != 200:
        return False, f"Zip download returned {zip_resp.status_code}"

    # 3. Extract top_lists.json from the zip
    try:
        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
            if "top_lists.json" not in zf.namelist():
                return False, "top_lists.json not found in artifact zip"
            raw = zf.read("top_lists.json")
            remote_data = json.loads(raw)
    except Exception as exc:
        return False, f"Zip extraction failed: {exc}"

    # 4. Compare timestamps — only write if remote is newer
    remote_ts_str = remote_data.get("generated_at", "")
    local_data = _load_top_lists()
    if local_data is not None:
        local_ts_str = local_data.get("generated_at", "")
        try:
            from datetime import datetime, timezone as _tz
            remote_ts = datetime.fromisoformat(remote_ts_str.replace("Z", "+00:00"))
            local_ts  = datetime.fromisoformat(local_ts_str.replace("Z", "+00:00"))
            if local_ts >= remote_ts:
                return True, "already_up_to_date"
        except Exception:
            pass  # unparseable timestamp → overwrite anyway

    # 5. Write atomically
    try:
        _TOP_LISTS.parent.mkdir(parents=True, exist_ok=True)
        tmp = _TOP_LISTS.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(_TOP_LISTS)
    except Exception as exc:
        return False, f"Write failed: {exc}"

    ticker_count = remote_data.get("ticker_count", "?")
    return True, f"Synced {ticker_count} tickers from artifact created {created_at[:16].replace('T', ' ')} UTC"
```

- [ ] **Step 2: Add the Sync button to `render()`**

In `render()`, find the existing button row (around line 244):

```python
    col_ref, col_ts = st.columns([1, 6])

    if col_ref.button("↻ Refresh", key="sp_refresh"):
        _load_top_lists.clear()
        st.session_state["sp_just_refreshed"] = True
        st.rerun()
```

Change it to add a third column with the Sync button:

```python
    col_ref, col_sync, col_ts = st.columns([1, 1, 5])

    if col_ref.button("↻ Refresh", key="sp_refresh"):
        _load_top_lists.clear()
        st.session_state["sp_just_refreshed"] = True
        st.rerun()

    _gh_pat_early = os.getenv("GH_PAT", "")
    if _gh_pat_early and col_sync.button("⬇ Sync", key="sp_sync", help="Download latest artifact from GitHub Actions"):
        with st.spinner("Downloading from GitHub…"):
            _ok, _msg = _sync_from_github(_gh_pat_early)
        if _ok and _msg == "already_up_to_date":
            st.toast("Already up to date — local file is newer or same.", icon="✅")
        elif _ok:
            _load_top_lists.clear()
            st.toast(_msg, icon="✅")
            st.rerun()
        else:
            st.error(f"Sync failed: {_msg}", icon="❌")
```

- [ ] **Step 3: Verify the page renders without errors**

Start the Streamlit app and navigate to Stock Picker. Confirm:
- "↻ Refresh" and "⬇ Sync" buttons appear side by side (Sync only if GH_PAT is set)
- Clicking Sync with GH_PAT set either syncs or shows "Already up to date"
- No Python exceptions in the terminal

If GH_PAT is not set in `.env`, the Sync button should not appear at all.

- [ ] **Step 4: Commit**

```
git add pages/6_Stock_Picker.py
git commit -m "feat(ui): add Sync from GitHub button to Stock Picker

Downloads the latest top-lists artifact from edgar_3x and writes
logs/top_lists.json atomically. Only overwrites if the artifact is
newer than the local file. Requires GH_PAT with actions:read scope.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Push and verify

- [ ] **Step 1: Push all commits**

```
git push origin main
```

- [ ] **Step 2: Verify CI passes**

Wait for the `CI` workflow to complete on GitHub Actions. All three jobs (sanity, smoke, test) must pass. The smoke test `test_streamlit_app_imports` confirms `_load_discovery` removal didn't break anything.

- [ ] **Step 3: Trigger `hybrid_pipeline` dry run**

```
# Using PowerShell with your PAT
$pat = (Get-Content .env | Where-Object { $_ -match "^GH_PAT=" }) -replace "^GH_PAT=", ""
$headers = @{ "Authorization" = "Bearer $pat"; "Accept" = "application/vnd.github+json"; "X-GitHub-Api-Version" = "2022-11-28" }
$body = @{ ref = "main"; inputs = @{ dry_run = "true"; force_refresh = "false" } } | ConvertTo-Json
Invoke-RestMethod -Uri "https://api.github.com/repos/Natarrr/regimetrader/actions/workflows/hybrid_pipeline.yml/dispatches" -Method Post -Headers $headers -Body $body -ContentType "application/json"
```

Expected: 204 (no content). Then watch the workflow run on GitHub Actions — the `quant` job should show "Shortlist (5 symbols): [...]" with tickers from `top_buys`, and the `claude` job should show `[DRY RUN] Would analyse NFLX — skipping API call` (or whichever tickers are current top buys).

- [ ] **Step 4: Trigger `edgar_3x` to test archive job**

Trigger `edgar_3x` with `force_regen=true` via the Stock Picker page or via API. After it completes (~40 min), check:

```
git pull
git log --oneline logs/archive/
```

Expected: one commit with message `chore(archive): snapshot 2026-05-19` and a file `logs/archive/2026-05-19_top_lists.json`.

---

## Self-Review Checklist

- [x] All four spec changes have tasks: dead code (Task 1), gitignore (Task 2), archive job (Task 3), hybrid_pipeline (Task 4), sync button (Task 5)
- [x] No TBD/TODO placeholders — all code is complete and runnable
- [x] Field names consistent throughout: `inst_score` (not `institutional_score`) matches what `build_prompt` reads via `quant_data.get("inst_score", 0)`
- [x] `_sync_from_github` uses `_TOP_LISTS` (already defined in the file as `_ROOT / "logs" / "top_lists.json"`) — no new path constant needed
- [x] `_load_top_lists()` is called inside `_sync_from_github` for timestamp comparison — this is safe since it's a `@st.cache_data` function that returns the cached or disk value
- [x] Archive job uses `dawidd6/action-download-artifact@v8` (same version as `daily_toplists_discord.yml`) — consistent
- [x] `permissions: contents: write` is at the job level in archive job, not workflow level — keeps other jobs read-only
