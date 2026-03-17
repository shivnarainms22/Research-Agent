"""Cluster papers into named themes using k-means + Claude."""
from __future__ import annotations

import json
import uuid
from datetime import datetime

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from core import token_tracker
from core.models import ThemeCluster
from knowledge.theme_store import save_theme, clear_themes
from knowledge.vector_store import get_collection
from knowledge.paper_store import get_paper

log = structlog.get_logger()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    stop=stop_after_attempt(3),
)
def _name_cluster(titles_and_abstracts: str) -> dict:
    client = _get_client()

    tool = {
        "name": "name_theme",
        "description": "Give a name and description to a cluster of research papers",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short 2-5 word name for this research theme"},
                "description": {"type": "string", "description": "1-2 sentence summary of what unifies these papers"},
            },
            "required": ["name", "description"],
        },
    }

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=512,
        temperature=0.3,
        system=[{
            "type": "text",
            "text": "You are a research librarian. Given a cluster of related papers, give the cluster a short descriptive name and a 1-2 sentence description of the common research theme.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": f"Papers in this cluster:\n\n{titles_and_abstracts}\n\nName this research theme.",
        }],
        tools=[tool],
        tool_choice={"type": "tool", "name": "name_theme"},
    )

    token_tracker.track("theme_clusterer", response.usage.input_tokens, response.usage.output_tokens)

    result = next((b.input for b in response.content if b.type == "tool_use"), None)
    return result or {"name": "Unnamed Cluster", "description": ""}


def cluster_themes() -> list[ThemeCluster]:
    """Run k-means clustering on paper embeddings and name each cluster with Claude."""
    try:
        from sklearn.cluster import KMeans
        import numpy as np
    except ImportError:
        log.error("theme_clusterer.missing_sklearn")
        return []

    collection = get_collection()
    try:
        data = collection.get(include=["embeddings", "metadatas"])
    except Exception as e:
        log.error("theme_clusterer.chroma_get_error", error=str(e))
        return []

    embeddings = data.get("embeddings")
    if embeddings is None:
        embeddings = []
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []

    if len(embeddings) < 5:
        log.info("theme_clusterer.insufficient_papers", count=len(embeddings))
        return []

    n_clusters = min(8, len(embeddings) // 5)
    if n_clusters < 2:
        n_clusters = 2

    X = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Group paper IDs by cluster
    cluster_paper_ids: dict[int, list[str]] = {}
    for idx, label in enumerate(labels):
        cluster_paper_ids.setdefault(int(label), []).append(ids[idx])

    clear_themes()

    themes = []
    for cluster_idx, paper_ids in cluster_paper_ids.items():
        # Build title+abstract summary for Claude (up to 10 papers)
        lines = []
        for pid in paper_ids[:10]:
            paper = get_paper(pid)
            if paper:
                lines.append(f"- {paper.title}: {paper.abstract[:200]}")

        if not lines:
            continue

        titles_text = "\n".join(lines)
        try:
            result = _name_cluster(titles_text)
        except Exception as e:
            log.error("theme_clusterer.name_error", cluster=cluster_idx, error=str(e))
            result = {"name": f"Theme {cluster_idx + 1}", "description": ""}

        theme = ThemeCluster(
            id=str(uuid.uuid4()),
            name=result["name"],
            description=result["description"],
            paper_ids=json.dumps(paper_ids),
            paper_count=len(paper_ids),
            updated_at=datetime.utcnow(),
        )
        save_theme(theme)
        themes.append(theme)
        log.info("theme_clusterer.cluster_named", name=result["name"], papers=len(paper_ids))

    log.info("theme_clusterer.complete", clusters=len(themes))
    return themes
