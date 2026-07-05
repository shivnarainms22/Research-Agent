"""Tests for core/notifier.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from config import settings
from core import notifier


def test_noop_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "")
    monkeypatch.setattr(settings, "telegram_chat_id", "")
    monkeypatch.setattr(settings, "notify_webhook_url", "")
    with patch("core.notifier.httpx.post") as mock_post:
        assert notifier.notify("t", "m") is False
    mock_post.assert_not_called()


def test_telegram_send(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "TOKEN")
    monkeypatch.setattr(settings, "telegram_chat_id", "123")
    monkeypatch.setattr(settings, "notify_webhook_url", "")
    with patch("core.notifier.httpx.post", return_value=MagicMock(raise_for_status=lambda: None)) as mock_post:
        assert notifier.notify("Title", "Body") is True
    url = mock_post.call_args[0][0]
    assert "api.telegram.org/botTOKEN/sendMessage" in url
    assert mock_post.call_args.kwargs["json"]["chat_id"] == "123"


def test_webhook_send(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "")
    monkeypatch.setattr(settings, "telegram_chat_id", "")
    monkeypatch.setattr(settings, "notify_webhook_url", "https://hook.example/x")
    with patch("core.notifier.httpx.post", return_value=MagicMock(raise_for_status=lambda: None)) as mock_post:
        assert notifier.notify("Title", "Body") is True
    assert mock_post.call_args[0][0] == "https://hook.example/x"


def test_never_raises_on_failure(monkeypatch):
    monkeypatch.setattr(settings, "notify_webhook_url", "https://hook.example/x")
    monkeypatch.setattr(settings, "telegram_bot_token", "")
    with patch("core.notifier.httpx.post", side_effect=RuntimeError("network down")):
        assert notifier.notify("t", "m") is False
