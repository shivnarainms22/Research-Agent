"""Helpers for parsing Claude Messages API responses."""
from __future__ import annotations


def first_text(message) -> str:
    """Return the first text block's text from a Messages response.

    Sonnet-5 may emit a thinking block before the text block, so `content[0].text`
    is unsafe. Tool-use responses are extracted separately (filtered by block type);
    this is only for free-text responses.
    """
    if not getattr(message, "content", None):
        return ""
    return next((b.text for b in message.content if getattr(b, "type", None) == "text"), "")
