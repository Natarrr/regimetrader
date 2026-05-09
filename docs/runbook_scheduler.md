# Runbook: DST-Aware Discord Scheduler

**Service:** `trigger-daily-discord` Cloud Function  
**Owner:** regime_trader pipeline  
**Last updated:** 2026-05-07

---

## Overview

The `daily_toplists_discord` GitHub Actions workflow must fire at **14:00 Europe/London** year-round. Because GitHub Actions cron uses UTC, a static cron cannot track London DST transitions automatically.

**Solution:** A Google Cloud Function (`cloud/scheduler/gcf_scheduler.py`) is triggered by Cloud Scheduler with `--time-zone=Europe/London`. Cloud Scheduler translates "14:00 London" to the correct UTC moment each day, handling BST↔GMT transitions automatically.

---

## Architecture

```
Cloud Scheduler (cron 0 14 * * *, Europe/London)
      │  OIDC POST
      ▼
Cloud Function: trigger-daily-discord
      │  POST /repos/{owner}/{repo}/actions/workflows/daily_toplists_discord.yml/dispatches
      ▼
GitHub Actions: daily_toplists_discord workflow
      │  downloads top-lists artifact from edgar_3x
      ▼
Discord webhook → #trading-alerts channel
```

---

## Deployment

**First-time setup:**

```bash
# 1. Create GCP project and enable APIs
gcloud services enable cloudfunctions.googleapis.com cloudscheduler.googleapis.com secretmanager.googleapis.com

# 2. Create service account
gcloud iam service-accounts create gcf-scheduler \
  --display-name="GCF Discord Scheduler"

# 3. Grant invoker role (so Scheduler can call the Function)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:gcf-scheduler@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudfunctions.invoker"

# 4. Grant Secret Manager access
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:gcf-scheduler@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# 5. Store GitHub PAT (needs 'workflow' scope)
echo -n "ghp_YOUR_TOKEN" | \
  gcloud secrets create GITHUB_TOKEN_SCHEDULER \
    --replication-policy=automatic \
    --data-file=-

# 6. Deploy function + scheduler
PROJECT_ID=my-gcp-project \
GITHUB_REPO=Natarrr/regimetrader \
  bash infra/gcf_deploy.sh
```

**Re-deploy after code change:**

```bash
PROJECT_ID=my-gcp-project GITHUB_REPO=Natarrr/regimetrader bash infra/gcf_deploy.sh
```

---

## DST Verification

After deployment, manually trigger the scheduler and check:

```bash
# Trigger immediately
gcloud scheduler jobs run daily-discord-london --location=europe-west1

# Check Cloud Function logs
gcloud functions logs read trigger-daily-discord \
  --gen2 --region=europe-west1 --limit=20

# Expected log line during BST (UTC+1, Apr–Oct):
# Triggered at London=2026-07-01T14:00:XX+01:00  DST=True  UTC+1

# Expected log line during GMT (UTC+0, Nov–Mar):
# Triggered at London=2026-12-01T14:00:XX+00:00  DST=False  UTC+0
```

To verify DST transitions specifically:
- Last Sunday in March: London transitions BST → Europe fires at 13:00 UTC
- Last Sunday in October: London transitions GMT → Europe fires at 14:00 UTC

---

## Secret Rotation

```bash
# Rotate GitHub PAT
echo -n "ghp_NEW_TOKEN" | \
  gcloud secrets versions add GITHUB_TOKEN_SCHEDULER --data-file=-

# The Cloud Function automatically uses the 'latest' version.
# No redeploy required.
```

---

## Monitoring

**Cloud Scheduler:** GCP Console → Cloud Scheduler → `daily-discord-london`
- Columns: Last run, Last result, Next run
- Expected last result: `Success`

**Cloud Function:** GCP Console → Cloud Functions → `trigger-daily-discord` → Logs
- Filter: `severity>=WARNING`
- Alert on: any log with `"status": "error"` or `"All retries"` pattern

**GitHub Actions:** GitHub → Actions → `daily_toplists_discord`
- Should show a run at ~14:00 London daily
- The `source_run_id` field in the job summary identifies GCF-triggered runs: `gcf-YYYYMMDDTHHMM`

---

## Incident Response

### Symptom: Discord message not received at 14:00

1. Check Cloud Scheduler last run status in GCP Console
2. If `Success`: check Cloud Function logs for dispatch errors (auth, 422, etc.)
3. If `Failed`: check if OIDC auth is broken (service account permissions)
4. Check GitHub Actions for `daily_toplists_discord` run — did it trigger?
5. If run triggered but Discord silent: check `discord-send-log-*` artifact in GitHub Actions

### Symptom: Wrong time (DST mismatch)

Verify Cloud Scheduler timezone:
```bash
gcloud scheduler jobs describe daily-discord-london --location=europe-west1 \
  | grep timeZone
# Expected: timeZone: Europe/London
```

If wrong, run `bash infra/gcf_deploy.sh` to restore correct config.

### Symptom: `401 Unauthorized` in Cloud Function logs

The GitHub PAT has expired or been revoked:
1. Generate a new PAT with `workflow` scope on GitHub
2. Rotate the secret (see Secret Rotation above)
3. Manually trigger the scheduler to verify: `gcloud scheduler jobs run ...`

### Symptom: `422 Unprocessable` in Cloud Function logs

The `ref` (branch) or workflow file does not exist:
1. Check `GITHUB_REF` env var in Cloud Function — should be `main`
2. Verify `daily_toplists_discord.yml` exists on that branch
3. Redeploy with correct `GITHUB_REF` if needed

---

## Rollback

If the GCF scheduler causes issues, fall back to the static cron in `daily_toplists_discord.yml`:

```yaml
# .github/workflows/daily_toplists_discord.yml
on:
  schedule:
    - cron: "0 13 * * *"   # BST: correct; GMT winter: 1 hour early
```

Update the cron manually at each DST transition, or pause the Cloud Scheduler job:

```bash
gcloud scheduler jobs pause daily-discord-london --location=europe-west1
```
