"""Orchestrates paper analysis, experiment extraction, and embedding."""
from __future__ import annotations

import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings
from core.models import Paper, PaperAnalysis, RunState
from core.state import save_state
from knowledge import paper_store, vector_store
from knowledge.experiment_store import save_experiment
from synthesis import paper_analyzer, experiment_extractor

log = structlog.get_logger()


def _passes_keyword_filter(paper: Paper) -> bool:
    text = f"{paper.title} {paper.abstract}".lower()
    matches = sum(1 for kw in settings.arxiv_keywords if kw.lower() in text)
    passes = matches >= settings.min_keyword_matches_to_analyze
    if not passes:
        log.info("synthesis.keyword_filter_skip", paper_id=paper.id, title=paper.title[:60])
    return passes


def _analyze_one(paper: Paper) -> tuple[str, PaperAnalysis | None, str | None]:
    """Fetch full text + analyze a single paper. Returns (paper_id, analysis, error)."""
    paper_id = paper.id
    try:
        # Fetch full text for arXiv papers that don't have it yet
        if paper.source == "arxiv" and not paper.full_text:
            from ingestion.fulltext_fetcher import fetch_arxiv_fulltext
            ft = fetch_arxiv_fulltext(paper.source_id)
            if ft:
                paper.full_text = ft
                paper_store.update_paper_full_text(paper.id, ft)
                log.info("synthesis.fulltext_fetched", paper_id=paper.id, chars=len(ft))

        analysis = paper_analyzer.analyze_paper(paper)
        return paper_id, analysis, None
    except Exception as e:
        return paper_id, None, str(e)


def run(state: RunState) -> list[str]:
    """Analyze new papers and generate experiments. Returns list of experiment IDs."""
    paper_ids = state.paper_ids_this_cycle

    # Fall back to all unanalyzed papers in DB (e.g. manually ingested)
    if not paper_ids:
        unanalyzed = paper_store.get_papers_by_status("fetched")
        paper_ids = [p.id for p in unanalyzed]

    if not paper_ids:
        log.info("synthesis.no_papers")
        return []

    # Collect papers that need analysis
    papers_to_analyze: list[Paper] = []
    for paper_id in paper_ids:
        paper = paper_store.get_paper(paper_id)
        if paper is None:
            continue
        existing = paper_store.get_analysis(paper_id)
        if existing:
            log.debug("synthesis.already_analyzed", paper_id=paper_id)
            continue
        if not _passes_keyword_filter(paper):
            paper_store.update_paper_status(paper_id, "done")
            continue
        papers_to_analyze.append(paper)

    # -----------------------------------------------------------------
    # Phase 1: Parallel analysis (no ChromaDB writes in workers)
    # -----------------------------------------------------------------
    analysis_results: dict[str, PaperAnalysis] = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_analyze_one, paper): paper for paper in papers_to_analyze}
        for future in as_completed(futures):
            pid, analysis, error = future.result()
            if error:
                log.error("synthesis.paper_failed", paper_id=pid, error=error)
            elif analysis is not None:
                paper_store.save_analysis(analysis)
                paper_store.update_paper_status(pid, "analyzed")
                paper_store.update_paper_full_text(pid, None)  # free storage
                log.info(
                    "synthesis.analyzed",
                    paper_id=pid,
                    novelty=analysis.novelty_score,
                    relevance=analysis.relevance_score,
                    difficulty=analysis.reproducibility_difficulty,
                )
                analysis_results[pid] = analysis

    # -----------------------------------------------------------------
    # Phase 2: Sequential — embed + extract experiments (ChromaDB is not thread-safe)
    # -----------------------------------------------------------------
    experiment_ids: list[str] = []

    for paper in papers_to_analyze:
        paper_id = paper.id
        analysis = analysis_results.get(paper_id)
        if analysis is None:
            continue

        # Skip low-relevance papers before experiment generation
        if analysis.relevance_score < settings.min_relevance_score_to_experiment:
            log.info(
                "synthesis.relevance_filter_skip",
                paper_id=paper_id,
                score=analysis.relevance_score,
            )
            continue

        try:
            # Check for contradictions; also determine if there's a direct contradiction
            has_direct_contradiction = False
            try:
                from knowledge import contradiction_detector
                from knowledge.contradiction_store import get_contradictions_for_paper
                contradiction_detector.check_new_paper(paper_id, analysis)
                contras = get_contradictions_for_paper(paper_id)
                has_direct_contradiction = any(c.severity == "direct" for c in contras)
            except Exception as e:
                log.warning("synthesis.contradiction_check_failed", paper_id=paper_id, error=str(e))

            # Embed into vector store (sequential — ChromaDB not thread-safe)
            vector_store.embed_paper(paper)

            # Extract experiments, passing contradiction flag
            experiments = experiment_extractor.extract_experiments(
                paper_id, analysis, has_direct_contradiction=has_direct_contradiction
            )
            for exp in experiments:
                save_experiment(exp)
                experiment_ids.append(exp.id)

        except Exception as e:
            log.error("synthesis.phase2_failed", paper_id=paper_id, error=str(e))

    state.experiment_ids_this_cycle.extend(experiment_ids)
    save_state(state)

    log.info("synthesis.complete", experiments_created=len(experiment_ids))
    return experiment_ids
