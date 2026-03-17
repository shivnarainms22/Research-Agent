"""NetworkX knowledge graph of paper relationships, serialized to JSON."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import structlog

from config import settings
from core.models import PaperAnalysis, parse_json_list

log = structlog.get_logger()

_GRAPH_PATH = None


def _graph_path() -> Path:
    return settings.data_dir / "knowledge_graph.json"


def load_graph() -> nx.DiGraph:
    path = _graph_path()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return nx.node_link_graph(data)
    return nx.DiGraph()


def save_graph(G: nx.DiGraph) -> None:
    data = nx.node_link_data(G)
    _graph_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_paper(G: nx.DiGraph, paper_id: str, title: str, contributions: list[str]) -> nx.DiGraph:
    G.add_node(paper_id, title=title, contributions=contributions)
    return G


def add_relationship(G: nx.DiGraph, from_id: str, to_id: str, relation: str = "related") -> nx.DiGraph:
    G.add_edge(from_id, to_id, relation=relation)
    return G


def rebuild(analyses: list[tuple[str, str, PaperAnalysis]]) -> nx.DiGraph:
    """Rebuild graph from list of (paper_id, title, analysis) tuples."""
    G = nx.DiGraph()
    for paper_id, title, analysis in analyses:
        contributions = parse_json_list(analysis.key_contributions)
        add_paper(G, paper_id, title, contributions)

    # Add edges between papers that share methods
    method_map: dict[str, list[str]] = {}
    for paper_id, _, analysis in analyses:
        for method in parse_json_list(analysis.methods_described):
            method_map.setdefault(method.lower()[:50], []).append(paper_id)

    for method, pids in method_map.items():
        if len(pids) > 1:
            for i, pid_a in enumerate(pids):
                for pid_b in pids[i + 1:]:
                    add_relationship(G, pid_a, pid_b, relation=f"shares_method:{method[:30]}")

    save_graph(G)
    log.info("knowledge_graph.rebuilt", nodes=G.number_of_nodes(), edges=G.number_of_edges())
    return G
