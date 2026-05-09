#!/usr/bin/env bash
# infra/gcf_deploy.sh
# Deploy the DST-aware Cloud Function + Cloud Scheduler job.
#
# Prerequisites:
#   gcloud auth login && gcloud config set project PROJECT_ID
#   Secret Manager secret GITHUB_TOKEN_SCHEDULER already created:
#     gcloud secrets create GITHUB_TOKEN_SCHEDULER --replication-policy=automatic
#     echo -n "ghp_..." | gcloud secrets versions add GITHUB_TOKEN_SCHEDULER --data-file=-
#
# Usage:
#   PROJECT_ID=my-gcp-project GITHUB_REPO=Natarrr/regimetrader bash infra/gcf_deploy.sh
#
# Optional overrides:
#   REGION          (default: europe-west1)
#   FUNCTION_NAME   (default: trigger-daily-discord)
#   SERVICE_ACCOUNT (default: gcf-scheduler@PROJECT_ID.iam.gserviceaccount.com)
#   GITHUB_REF      (default: main)
#   TIME_TOLERANCE_MINUTES (default: 30)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
GITHUB_REPO="${GITHUB_REPO:?Set GITHUB_REPO (owner/repo)}"
REGION="${REGION:-europe-west1}"
FUNCTION_NAME="${FUNCTION_NAME:-trigger-daily-discord}"
GITHUB_REF="${GITHUB_REF:-main}"
TIME_TOLERANCE_MINUTES="${TIME_TOLERANCE_MINUTES:-30}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-gcf-scheduler@${PROJECT_ID}.iam.gserviceaccount.com}"
SCHEDULER_JOB="daily-discord-london"

echo "=== Deploying ${FUNCTION_NAME} to ${PROJECT_ID}/${REGION} ==="

# ── 1. Deploy Cloud Function ──────────────────────────────────────────────────
gcloud functions deploy "${FUNCTION_NAME}" \
  --gen2 \
  --runtime=python311 \
  --region="${REGION}" \
  --source=cloud/scheduler \
  --entry-point=trigger_daily_discord \
  --trigger-http \
  --allow-unauthenticated=false \
  --service-account="${SERVICE_ACCOUNT}" \
  --memory=256Mi \
  --timeout=60s \
  --set-env-vars="GITHUB_REPO=${GITHUB_REPO},GITHUB_REF=${GITHUB_REF},TIME_TOLERANCE_MINUTES=${TIME_TOLERANCE_MINUTES}" \
  --set-secrets="GITHUB_TOKEN_SCHEDULER=GITHUB_TOKEN_SCHEDULER:latest" \
  --project="${PROJECT_ID}"

# ── 2. Retrieve Cloud Function URL ───────────────────────────────────────────
FUNCTION_URL=$(gcloud functions describe "${FUNCTION_NAME}" \
  --gen2 \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(serviceConfig.uri)")

echo "Function URL: ${FUNCTION_URL}"

# ── 3. Create / update Cloud Scheduler job ───────────────────────────────────
# Cloud Scheduler handles DST automatically with --time-zone=Europe/London.
# The cron "0 14 * * *" fires at exactly 14:00 London local time year-round.
if gcloud scheduler jobs describe "${SCHEDULER_JOB}" \
     --location="${REGION}" \
     --project="${PROJECT_ID}" &>/dev/null; then
  echo "Updating existing scheduler job '${SCHEDULER_JOB}' ..."
  gcloud scheduler jobs update http "${SCHEDULER_JOB}" \
    --location="${REGION}" \
    --schedule="0 14 * * *" \
    --time-zone="Europe/London" \
    --uri="${FUNCTION_URL}" \
    --http-method=POST \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
    --oidc-token-audience="${FUNCTION_URL}" \
    --attempt-deadline=60s \
    --max-retry-attempts=2 \
    --project="${PROJECT_ID}"
else
  echo "Creating scheduler job '${SCHEDULER_JOB}' ..."
  gcloud scheduler jobs create http "${SCHEDULER_JOB}" \
    --location="${REGION}" \
    --schedule="0 14 * * *" \
    --time-zone="Europe/London" \
    --uri="${FUNCTION_URL}" \
    --http-method=POST \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
    --oidc-token-audience="${FUNCTION_URL}" \
    --attempt-deadline=60s \
    --max-retry-attempts=2 \
    --project="${PROJECT_ID}"
fi

echo ""
echo "=== Done ==="
echo "Scheduler job '${SCHEDULER_JOB}' fires at 14:00 Europe/London daily."
echo "Manual trigger: gcloud scheduler jobs run ${SCHEDULER_JOB} --location=${REGION}"
echo ""
echo "DST verification:"
echo "  BST (Apr-Oct): Cloud Scheduler fires at 13:00 UTC → GitHub dispatch"
echo "  GMT (Nov-Mar): Cloud Scheduler fires at 14:00 UTC → GitHub dispatch"
echo "Both arrive at 14:00 London local time."
