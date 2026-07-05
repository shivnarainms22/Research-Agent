"""Tests for knowledge/lesson_store.py."""
from __future__ import annotations

from knowledge.lesson_store import save_lesson, get_recent_lessons


def test_save_and_retrieve_recent(in_memory_engine):
    save_lesson("e1", "first lesson", category="repair", paper_id="p1")
    save_lesson("e2", "second lesson", category="failure")
    lessons = get_recent_lessons(limit=5)
    texts = {l.text for l in lessons}
    assert texts == {"first lesson", "second lesson"}


def test_empty_text_is_ignored(in_memory_engine):
    save_lesson("e1", "")
    assert get_recent_lessons() == []


def test_limit_respected(in_memory_engine):
    for i in range(10):
        save_lesson(f"e{i}", f"lesson {i}")
    assert len(get_recent_lessons(limit=3)) == 3
