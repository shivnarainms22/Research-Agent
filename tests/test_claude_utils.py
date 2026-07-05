"""Tests for core/claude_utils.first_text."""
from __future__ import annotations

from types import SimpleNamespace

from core.claude_utils import first_text


def _block(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def test_skips_thinking_block():
    """Sonnet-5 puts a thinking block first — first_text must return the text block."""
    msg = SimpleNamespace(content=[
        _block("thinking", thinking="pondering..."),
        _block("text", text="the answer"),
    ])
    assert first_text(msg) == "the answer"


def test_plain_text_first():
    msg = SimpleNamespace(content=[_block("text", text="hello")])
    assert first_text(msg) == "hello"


def test_empty_content():
    assert first_text(SimpleNamespace(content=[])) == ""
    assert first_text(SimpleNamespace(content=None)) == ""


def test_no_text_block():
    msg = SimpleNamespace(content=[_block("thinking", thinking="only thinking")])
    assert first_text(msg) == ""
