"""Find and fetch a paper's official code repository to ground experiment codegen."""
from __future__ import annotations

import re

import httpx
import structlog

from core.models import Paper

log = structlog.get_logger()

_GITHUB_RE = re.compile(r"https?://github\.com/([\w-]+)/([\w.-]+)", re.IGNORECASE)
_README_MAX_CHARS = 8000
_TIMEOUT = 15.0


def find_github_repo(paper: Paper) -> str | None:
    """Return 'owner/repo' from the paper's own text, else Papers With Code lookup."""
    text = f"{paper.abstract or ''} {paper.full_text or ''}"
    m = _GITHUB_RE.search(text)
    if m:
        repo = m.group(2).removesuffix(".git").rstrip(".")
        return f"{m.group(1)}/{repo}"

    if paper.source == "arxiv":
        arxiv_id = re.sub(r"v\d+$", "", paper.source_id)
        try:
            r = httpx.get(
                f"https://paperswithcode.com/api/v1/papers/arxiv:{arxiv_id}/repositories/",
                timeout=_TIMEOUT, follow_redirects=True,
            )
            if r.status_code == 200:
                repos = r.json().get("results", [])
                repos.sort(key=lambda x: not x.get("is_official", False))  # official first
                for entry in repos:
                    m = _GITHUB_RE.search(entry.get("url", ""))
                    if m:
                        return f"{m.group(1)}/{m.group(2).removesuffix('.git')}"
        except Exception as e:
            log.debug("repo_fetcher.pwc_lookup_failed", paper_id=paper.id, error=str(e))
    return None


def fetch_readme(repo: str) -> str | None:
    """Fetch a repo's README as raw text (public repos, unauthenticated)."""
    try:
        r = httpx.get(
            f"https://api.github.com/repos/{repo}/readme",
            headers={"Accept": "application/vnd.github.raw+json"},
            timeout=_TIMEOUT, follow_redirects=True,
        )
        if r.status_code == 200:
            return r.text[:_README_MAX_CHARS]
    except Exception as e:
        log.debug("repo_fetcher.readme_failed", repo=repo, error=str(e))
    return None


def get_repo_context(paper: Paper) -> str:
    """Best-effort README excerpt for codegen grounding; '' when unavailable."""
    try:
        repo = find_github_repo(paper)
        if not repo:
            return ""
        readme = fetch_readme(repo)
        if not readme:
            return ""
        log.info("repo_fetcher.found", paper_id=paper.id, repo=repo, chars=len(readme))
        return (
            f"Official repository: https://github.com/{repo}\n"
            f"README excerpt:\n{readme}"
        )
    except Exception:
        return ""
