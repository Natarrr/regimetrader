"""monitoring/ — canary metrics export, Slack alerting, threshold checking."""
from .evaluate import evaluate
from .slack_notifier import send_slack_alert

__all__ = ["evaluate", "send_slack_alert"]
