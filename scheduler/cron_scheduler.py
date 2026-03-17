"""APScheduler job definitions with SQLite jobstore."""
from __future__ import annotations

import structlog
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config import settings
from scheduler.pipeline_runner import run_cycle, run_experiment_poll

log = structlog.get_logger()


def _ingestion_job():
    log.info("scheduler.ingestion_job_start")
    try:
        run_cycle(days_back=1)
    except Exception as e:
        log.error("scheduler.ingestion_job_error", error=str(e))


def _experiment_poll_job():
    log.info("scheduler.experiment_poll_start")
    try:
        run_experiment_poll()
    except Exception as e:
        log.error("scheduler.experiment_poll_error", error=str(e))


def _weekly_report_job():
    log.info("scheduler.weekly_report_start")
    try:
        from core.state import find_incomplete_states, new_state, mark_complete
        from datetime import datetime
        from reporting import report_generator

        incomplete = find_incomplete_states()
        if incomplete:
            state = incomplete[-1]
        else:
            state = new_state(f"weekly_{datetime.utcnow().strftime('%Y%m%d')}")

        report_generator.generate(state, report_type="weekly")
    except Exception as e:
        log.error("scheduler.weekly_report_error", error=str(e))


def _gap_finder_job():
    log.info("scheduler.gap_finder_start")
    try:
        from datetime import datetime
        from knowledge.gap_finder import find_gaps
        cycle_id = f"weekly_{datetime.utcnow().strftime('%Y%m%d')}"
        find_gaps(cycle_id)
    except Exception as e:
        log.error("scheduler.gap_finder_error", error=str(e))


def _theme_clusterer_job():
    log.info("scheduler.theme_clusterer_start")
    try:
        from knowledge.theme_clusterer import cluster_themes
        cluster_themes()
    except Exception as e:
        log.error("scheduler.theme_clusterer_error", error=str(e))


def _knowledge_graph_rebuild_job():
    log.info("scheduler.kg_rebuild_start")
    try:
        from knowledge.paper_store import get_all_papers, get_analysis
        from synthesis.knowledge_graph import rebuild

        papers = get_all_papers(limit=5000)
        triples = []
        for paper in papers:
            analysis = get_analysis(paper.id)
            if analysis:
                triples.append((paper.id, paper.title, analysis))

        rebuild(triples)
    except Exception as e:
        log.error("scheduler.kg_rebuild_error", error=str(e))


def create_scheduler() -> BlockingScheduler:
    db_url = f"sqlite:///{settings.db_path}"
    jobstores = {"default": SQLAlchemyJobStore(url=db_url)}
    executors = {"default": ThreadPoolExecutor(max_workers=1)}
    job_defaults = {"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600}

    scheduler = BlockingScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
    )

    # Twice-daily ingestion + full cycle
    for hour in settings.ingestion_hours:
        scheduler.add_job(
            _ingestion_job,
            trigger=CronTrigger(hour=hour, jitter=300),
            id=f"ingestion_{hour}h",
            replace_existing=True,
        )

    # Experiment poller every N minutes
    scheduler.add_job(
        _experiment_poll_job,
        trigger=IntervalTrigger(minutes=settings.experiment_poll_minutes),
        id="experiment_poll",
        replace_existing=True,
    )

    # Gap finder — Sunday 21:00 (1hr before weekly report)
    scheduler.add_job(
        _gap_finder_job,
        trigger=CronTrigger(day_of_week="sun", hour=21),
        id="gap_finder",
        replace_existing=True,
    )

    # Weekly report on Sunday 22:00
    scheduler.add_job(
        _weekly_report_job,
        trigger=CronTrigger(day_of_week="sun", hour=22),
        id="weekly_report",
        replace_existing=True,
    )

    # Knowledge graph rebuild Wednesday 03:00
    scheduler.add_job(
        _knowledge_graph_rebuild_job,
        trigger=CronTrigger(day_of_week="wed", hour=3),
        id="kg_rebuild",
        replace_existing=True,
    )

    # Theme clusterer — Wednesday 03:30
    scheduler.add_job(
        _theme_clusterer_job,
        trigger=CronTrigger(day_of_week="wed", hour=3, minute=30),
        id="theme_clusterer",
        replace_existing=True,
    )

    log.info("scheduler.created", jobs=len(scheduler.get_jobs()))
    return scheduler


def start():
    scheduler = create_scheduler()
    log.info("scheduler.starting")
    scheduler.start()
