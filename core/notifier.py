"""Best-effort push notifications (Telegram or generic webhook).

Unconfigured is a silent no-op; failures are swallowed so a notification can
never break the pipeline.
"""
from __future__ import annotations

import httpx
import structlog

from config import settings

log = structlog.get_logger()

_TIMEOUT = 10.0


def is_configured() -> bool:
    return bool(
        (settings.telegram_bot_token and settings.telegram_chat_id)
        or settings.notify_webhook_url
    )


def notify(title: str, message: str) -> bool:
    """Send a notification. Returns True if a channel accepted it, else False."""
    if not is_configured():
        return False
    try:
        if settings.telegram_bot_token and settings.telegram_chat_id:
            r = httpx.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                json={"chat_id": settings.telegram_chat_id, "text": f"*{title}*\n{message}",
                      "parse_mode": "Markdown"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            return True
        if settings.notify_webhook_url:
            r = httpx.post(
                settings.notify_webhook_url,
                json={"title": title, "message": message},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            return True
    except Exception as e:
        log.warning("notifier.send_failed", error=str(e))
    return False
