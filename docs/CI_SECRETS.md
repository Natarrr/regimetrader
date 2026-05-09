# CI Secrets Setup

## Overview

`scripts/check_secrets.py` verifies that required API keys are present in the
CI environment before the test suite runs. Values are **never printed** —
only `True` / `False` presence flags are shown.

## Required secrets

| Secret name | Purpose | Absence effect |
| ----------- | ------- | -------------- |
| `FMP_API_KEY` | Financial Modeling Prep stable/profile calls | `get_profile()` returns `None` |
| `ALPACA_API_KEY` | Alpaca broker paper/live trading | Broker functions disabled |
| `ALPACA_SECRET_KEY` | Alpaca broker auth complement | Broker functions disabled |

## Optional secrets (reported, not enforced)

| Secret name | Purpose |
| ----------- | ------- |
| `POLYGON_API_KEY` | Polygon.io market data (future use) |
| `ANTHROPIC_API_KEY` | Claude LLM integration |

## Adding secrets to GitHub

1. Go to **GitHub → your repo → Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Name: `FMP_API_KEY` (exact match, case-sensitive)
4. Value: paste the key — never commit it to the repo
5. Repeat for each secret

## Running the check locally

```bash
# With secrets in your shell environment or .env
source .env          # or: export FMP_API_KEY=sk-...
python scripts/check_secrets.py
```

Expected output (all present):

```
────────────────────────────────────
CI secret presence check
────────────────────────────────────
  FMP_API_KEY            present: True   [REQUIRED]  ✓
  ALPACA_API_KEY         present: True   [REQUIRED]  ✓
  ALPACA_SECRET_KEY      present: True   [REQUIRED]  ✓
  POLYGON_API_KEY        present: False  [optional]
  ANTHROPIC_API_KEY      present: True   [optional]
────────────────────────────────────

✓ All 3 required secrets present.
```

Expected output (one missing):

```
  ALPACA_API_KEY         present: False  [REQUIRED]  ✗ MISSING

✗ 1 required secret(s) missing: ALPACA_API_KEY
  Set them in GitHub → Settings → Secrets → Actions
```

## CI behaviour

The `check_secrets.py` step runs with `continue-on-error: true` so that fork
PRs (which have no access to secrets) are not hard-blocked. Change to
`continue-on-error: false` in `.github/workflows/ci.yml` to enforce a hard gate
once all secrets are registered.

## Security notes

- Secrets are injected as environment variables via the GitHub Actions
  `env:` block using `${{ secrets.X || '' }}`. The `|| ''` expression ensures
  the variable is defined (empty string) even for fork PRs.
- The check script uses `os.environ.get(key, "").strip()` — presence is defined
  as a non-empty string. An empty `FMP_API_KEY=""` is treated as absent.
- Never log or print the value of any secret. `SecretMaskFilter` in
  `regime_trader/utils/logging_cfg.py` provides a second line of defence.
