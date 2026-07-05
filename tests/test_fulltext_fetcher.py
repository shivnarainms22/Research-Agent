"""Tests for ingestion/fulltext_fetcher.py PDF fallback."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from ingestion import fulltext_fetcher


def test_falls_back_to_pdf_when_html_missing():
    """A 404 on the HTML endpoint routes to the PDF extractor."""
    html_resp = MagicMock(status_code=404)
    with (
        patch("ingestion.fulltext_fetcher.httpx.get", return_value=html_resp),
        patch("ingestion.fulltext_fetcher._fetch_from_pdf", return_value="pdf text") as mock_pdf,
    ):
        out = fulltext_fetcher.fetch_arxiv_fulltext("2401.00001")
    assert out == "pdf text"
    mock_pdf.assert_called_once()


def test_html_success_skips_pdf():
    """Usable HTML text short-circuits before the PDF fallback."""
    html_resp = MagicMock(status_code=200, text="<html><body><p>hello world body</p></body></html>")
    with (
        patch("ingestion.fulltext_fetcher.httpx.get", return_value=html_resp),
        patch("ingestion.fulltext_fetcher._fetch_from_pdf") as mock_pdf,
    ):
        out = fulltext_fetcher.fetch_arxiv_fulltext("2401.00001")
    assert out and "hello world body" in out
    mock_pdf.assert_not_called()


def test_pdf_fallback_caps_length():
    """PDF text is truncated to max_chars."""
    long_text = "x" * 100
    pdf_resp = MagicMock(status_code=200, content=b"%PDF-fake")
    fake_doc = [MagicMock(**{"get_text.return_value": long_text})]
    with (
        patch("ingestion.fulltext_fetcher.httpx.get", return_value=pdf_resp),
        patch("fitz.open", return_value=fake_doc),
    ):
        out = fulltext_fetcher._fetch_from_pdf("2401.00001", max_chars=10)
    assert out == "x" * 10
