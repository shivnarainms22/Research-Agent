"""Orchestrates paper analysis, experiment extraction, and embedding."""
from __future__ import annotations

import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings
from core.models import Experiment, Paper, PaperAnalysis, RunState
from core.state import save_state
from knowledge import contradiction_detector, contradiction_store, paper_store, vector_store
from knowledge.experiment_store import save_experiment
from synthesis import paper_analyzer, experiment_extractor

log = structlog.get_logger()

# Below this many papers, direct API calls beat batch turnaround time.
_MIN_PAPERS_FOR_BATCH = 3


def _passes_keyword_filter(paper: Paper) -> bool:
    text = f"{paper.title} {paper.abstract}".lower()
    matches = sum(1 for kw in settings.arxiv_keywords if kw.lower() in text)
    passes = matches >= settings.min_keyword_matches_to_analyze
    if not passes:
        log.info("synthesis.keyword_filter_skip", paper_id=paper.id, title=paper.title[:60])
    return passes


def _fetch_fulltext(paper: Paper) -> None:
    """Fetch arXiv full text into memory for the analysis call (not persisted)."""
    if paper.source != "arxiv" or paper.full_text:
        return
    try:
        from ingestion.fulltext_fetcher import fetch_arxiv_fulltext
        ft = fetch_arxiv_fulltext(paper.source_id)
        if ft:
            paper.full_text = ft
            log.info("synthesis.fulltext_fetched", paper_id=paper.id, chars=len(ft))
    except Exception as e:
        log.warning("synthesis.fulltext_failed", paper_id=paper.id, error=str(e))


def _extract_one(paper: Paper, analysis: PaperAnalysis) -> list[Experiment]:
    """Contradiction check + experiment extraction for one paper (thread-safe)."""
    try:
        has_direct_contradiction = False
        try:
            contradiction_detector.check_new_paper(paper.id, analysis)
            contras = contradiction_store.get_contradictions_for_paper(paper.id)
            has_direct_contradiction = any(c.severity == "direct" for c in contras)
        except Exception as e:
            log.warning("synthesis.contradiction_check_failed", paper_id=paper.id, error=str(e))

        return experiment_extractor.extract_experiments(
            paper.id, analysis, has_direct_contradiction=has_direct_contradiction, paper=paper
        )
    except Exception as e:
        log.error("synthesis.phase2_failed", paper_id=paper.id, error=str(e))
        return []


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
    # Phase 1: fetch full text in parallel, then analyze via the Batch
    # API (50% cheaper); small counts use direct calls for faster turnaround.
    # -----------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(_fetch_fulltext, papers_to_analyze))

    analysis_results: dict[str, PaperAnalysis] = {}
    if len(papers_to_analyze) >= _MIN_PAPERS_FOR_BATCH:
        analysis_results = paper_analyzer.analyze_papers_batch(papers_to_analyze)
    else:
        for paper in papers_to_analyze:
            try:
                analysis_results[paper.id] = paper_analyzer.analyze_paper(paper)
            except Exception as e:
                log.error("synthesis.paper_failed", paper_id=paper.id, error=str(e))

    for pid, analysis in analysis_results.items():
        paper_store.save_analysis(analysis)
        paper_store.update_paper_status(pid, "analyzed")
        log.info(
            "synthesis.analyzed",
            paper_id=pid,
            novelty=analysis.novelty_score,
            relevance=analysis.relevance_score,
            difficulty=analysis.reproducibility_difficulty,
        )

    # -----------------------------------------------------------------
    # Phase 2a: embed relevant papers (sequential — ChromaDB writes are
    # not thread-safe)
    # -----------------------------------------------------------------
    relevant: list[tuple[Paper, PaperAnalysis]] = []
    for paper in papers_to_analyze:
        analysis = analysis_results.get(paper.id)
        if analysis is None:
            continue
        if analysis.relevance_score < settings.min_relevance_score_to_experiment:
            log.info(
                "synthesis.relevance_filter_skip",
                paper_id=paper.id,
                score=analysis.relevance_score,
            )
            continue
        vector_store.embed_paper(paper)
        relevant.append((paper, analysis))

    # -----------------------------------------------------------------
    # Phase 2b: contradiction checks + experiment extraction in parallel
    # (Claude calls — the slow part); DB writes happen in the main thread.
    # -----------------------------------------------------------------
    experiment_ids: list[str] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_extract_one, p, a) for p, a in relevant]
        for future in as_completed(futures):
            for exp in future.result():
                save_experiment(exp)
                experiment_ids.append(exp.id)

    state.experiment_ids_this_cycle.extend(experiment_ids)
    save_state(state)

    log.info("synthesis.complete", experiments_created=len(experiment_ids))
    return experiment_ids
