"""CLI entrypoint for the Autonomous Research Agent."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import os
import structlog
import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

# Force UTF-8 output on Windows so Rich unicode characters render correctly
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(name="research-agent", help="Autonomous Research System")
console = Console()
log = structlog.get_logger()


def _configure_logging():
    import logging
    from config import settings

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )


@app.command()
def setup():
    """Initialize database, build Docker image, validate API keys."""
    _configure_logging()
    console.print("[bold blue]Setting up Research Agent...[/bold blue]")

    # 1. Init directories and DB
    from config import ensure_dirs, settings
    ensure_dirs()
    console.print(f"  [green]✓[/green] Data directories created at {settings.data_dir}")

    from core.database import init_db, init_chroma
    init_db()
    console.print("  [green]✓[/green] SQLite database initialized (WAL mode)")

    init_chroma()
    console.print("  [green]✓[/green] ChromaDB initialized")

    # 2. Validate API key
    if settings.anthropic_api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            # Minimal test call
            client.messages.create(
                model=settings.claude_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            console.print(f"  [green]✓[/green] Claude API key valid ({settings.claude_model})")
        except Exception as e:
            console.print(f"  [red]✗[/red] Claude API error: {e}")
    else:
        console.print("  [yellow]![/yellow] ANTHROPIC_API_KEY not set — add to .env")

    # 3. Build Docker image
    try:
        import docker
        client = docker.from_env()
        console.print("  Building Docker sandbox image (this may take a few minutes)...")
        docker_dir = Path(__file__).parent / "docker"
        client.images.build(
            path=str(docker_dir),
            dockerfile="Dockerfile.sandbox",
            tag="research-sandbox:latest",
            rm=True,
        )
        console.print("  [green]✓[/green] Docker sandbox image built")
    except Exception as e:
        console.print(f"  [yellow]![/yellow] Docker image build skipped: {e}")

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("Run [cyan]uv run python main.py run[/cyan] to start a pipeline cycle.")


@app.command()
def run(
    days_back: int = typer.Option(1, "--days", "-d", help="Days of papers to fetch"),
):
    """Run one full pipeline cycle immediately."""
    _configure_logging()
    console.print("[bold blue]Starting pipeline cycle...[/bold blue]")

    from core.database import init_db, init_chroma
    init_db()
    init_chroma()

    from scheduler.pipeline_runner import run_cycle
    try:
        state = run_cycle(days_back=days_back)
        console.print(f"\n[bold green]Cycle complete![/bold green] ID: {state.cycle_id}")
        console.print(f"  Papers: {len(state.paper_ids_this_cycle)}")
        console.print(f"  Experiments: {len(state.experiment_ids_this_cycle)}")
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """Show current cycle state and pending experiments."""
    _configure_logging()
    from config import settings
    from core.state import find_incomplete_states
    from knowledge.experiment_store import get_experiments_by_status
    from knowledge.paper_store import get_all_papers

    papers = get_all_papers(limit=10000)
    console.print(f"\n[bold]System Status[/bold]")
    console.print(f"  Total papers in DB: {len(papers)}")

    for status_name in ["pending_review", "pending", "running", "completed", "failed", "skipped"]:
        exps = get_experiments_by_status(status_name)
        console.print(f"  Experiments [{status_name}]: {len(exps)}")

    incomplete = find_incomplete_states()
    if incomplete:
        console.print(f"\n[yellow]Incomplete cycles: {len(incomplete)}[/yellow]")
        for s in incomplete:
            console.print(f"  {s.cycle_id} — stage: {s.current_stage}")
    else:
        console.print("\n[green]No incomplete cycles.[/green]")

    # List recent reports
    reports_dir = settings.reports_dir
    if reports_dir.exists():
        reports = sorted(reports_dir.glob("*.md"), reverse=True)[:5]
        if reports:
            console.print(f"\n[bold]Recent Reports:[/bold]")
            for r in reports:
                console.print(f"  {r.name}")


@app.command()
def report(
    report_type: str = typer.Option("weekly", "--type", "-t"),
):
    """Force generate a research report for the most recent cycle."""
    _configure_logging()
    from core.database import init_db
    from core.state import find_incomplete_states, new_state
    from datetime import datetime
    from reporting import report_generator

    init_db()

    incomplete = find_incomplete_states()
    if incomplete:
        state = incomplete[-1]
    else:
        state = new_state(f"manual_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

    r = report_generator.generate(state, report_type=report_type)
    console.print(f"[green]Report generated:[/green] {r.id}")
    console.print(f"  Saved to: data/reports/{state.cycle_id}_{report_type}.md")


@app.command()
def ingest(
    source: str = typer.Option("all", "--source", "-s", help="arxiv|semantic_scholar|substack|all"),
    days: int = typer.Option(1, "--days", "-d"),
):
    """Manual ingestion from specified source."""
    _configure_logging()
    from core.database import init_db
    from core.models import Paper
    from sqlmodel import Session
    from core.database import get_engine
    from ingestion.deduplicator import deduplicate
    from config import settings

    init_db()

    papers: list[Paper] = []

    if source in ("arxiv", "all"):
        from ingestion import arxiv_client
        papers.extend(arxiv_client.fetch_papers(days_back=days))

    if source in ("semantic_scholar", "s2", "all"):
        from ingestion import semantic_scholar_client
        papers.extend(semantic_scholar_client.fetch_papers(days_back=days))

    if source in ("substack", "all"):
        from ingestion import substack_scraper
        papers.extend(substack_scraper.fetch_papers(days_back=max(days, 7)))

    new_papers = deduplicate(papers)[: settings.max_papers_per_cycle]
    with Session(get_engine(), expire_on_commit=False) as session:
        for p in new_papers:
            session.add(p)
        session.commit()

    console.print(f"[green]Ingested {len(new_papers)} new papers[/green] from {source}")

    # Write a summary JSON so the dashboard "Recent Cycles" table shows this run
    from datetime import datetime
    cycle_id = f"ingest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    summary = {
        "cycle_id": cycle_id,
        "started_at": datetime.utcnow().isoformat(),
        "current_stage": "done",
        "completed_stages": ["ingestion"],
        "paper_ids_this_cycle": [p.id for p in new_papers],
        "experiment_ids_this_cycle": [],
        "error_log": [],
        "is_complete": True,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    (settings.state_dir / f"{cycle_id}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )


@app.command()
def metrics():
    """Show token usage, cost estimates, and pipeline statistics."""
    _configure_logging()
    from config import settings
    from knowledge.paper_store import get_all_papers
    from knowledge.experiment_store import get_experiments_by_status, get_completed_results
    from collections import Counter

    # Paper stats
    papers = get_all_papers(limit=10000)
    source_counts = Counter(p.source for p in papers)
    status_counts = Counter(p.status for p in papers)
    console.print("\n[bold]Papers[/bold]")
    console.print(f"  Total: {len(papers)}")
    for src, cnt in sorted(source_counts.items()):
        console.print(f"  source={src}: {cnt}")
    for st, cnt in sorted(status_counts.items()):
        console.print(f"  status={st}: {cnt}")

    # Experiment stats
    console.print("\n[bold]Experiments[/bold]")
    completed = get_experiments_by_status("completed")
    all_results = get_completed_results(limit=1000)
    total_runtime = sum(r.runtime_seconds for r in all_results)
    avg_runtime = total_runtime / max(len(all_results), 1)
    for st in ["pending_review", "pending", "running", "completed", "failed", "skipped"]:
        exps = get_experiments_by_status(st)
        console.print(f"  status={st}: {len(exps)}")
    success_rate = round(len(completed) / max(len(completed) + len(get_experiments_by_status("failed")), 1) * 100, 1)
    console.print(f"  Success rate: {success_rate}%")
    console.print(f"  Avg runtime (completed): {avg_runtime:.1f}s")

    # Recent cycles with token usage
    console.print("\n[bold]Recent Cycles (last 5)[/bold]")
    state_files = sorted(settings.state_dir.glob("*.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    for sf in state_files:
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            cycle_id = data.get("cycle_id", sf.stem)
            stage = data.get("current_stage", "?")
            inp = data.get("total_input_tokens", 0)
            out = data.get("total_output_tokens", 0)
            cost = inp * 3 / 1_000_000 + out * 15 / 1_000_000
            console.print(f"  {cycle_id} | stage={stage} | in={inp:,} out={out:,} | est. cost=${cost:.4f}")
        except Exception:
            console.print(f"  {sf.name} (unreadable)")

    # Token usage history by module (from persistent DB)
    try:
        from core.database import init_db
        init_db()
        from knowledge.token_log_store import get_module_totals
        module_totals = get_module_totals()
        if module_totals:
            console.print("\n[bold]Token Usage History (all-time by module)[/bold]")
            for mod, data in sorted(module_totals.items(), key=lambda x: -x[1]["cost_usd"]):
                console.print(
                    f"  {mod}: in={data['input_tokens']:,} out={data['output_tokens']:,} "
                    f"cost=${data['cost_usd']:.4f}"
                )
            total_cost = sum(d["cost_usd"] for d in module_totals.values())
            total_in = sum(d["input_tokens"] for d in module_totals.values())
            total_out = sum(d["output_tokens"] for d in module_totals.values())
            console.print(
                f"  [bold]TOTAL[/bold]: in={total_in:,} out={total_out:,} cost=${total_cost:.4f}"
            )
    except Exception:
        pass

    # Reports
    reports_dir = settings.reports_dir
    report_count = len(list(reports_dir.glob("*.md"))) if reports_dir.exists() else 0
    console.print(f"\n[bold]Reports generated:[/bold] {report_count}")


@app.command()
def review(
    approve_all: bool = typer.Option(False, "--approve-all", help="Approve all pending_review experiments non-interactively"),
    auto_threshold: Optional[float] = typer.Option(None, "--auto-threshold", help="Auto-approve experiments whose paper novelty score meets this threshold"),
):
    """Interactive review queue: approve, reject, or edit pending_review experiments."""
    _configure_logging()
    from core.database import init_db
    from knowledge.experiment_store import (
        get_experiments_by_status,
        update_experiment_status,
        update_experiment_hypothesis,
    )
    from knowledge.paper_store import get_paper, get_analysis

    init_db()

    pending = get_experiments_by_status("pending_review")
    if not pending:
        console.print("[green]No experiments pending review.[/green]")
        return

    approved = rejected = skipped_count = 0
    total = len(pending)

    # Non-interactive approve-all mode
    if approve_all:
        for exp in pending:
            update_experiment_status(exp.id, "pending")
            approved += 1
        console.print(
            f"\n[bold]Review complete:[/bold] "
            f"[green]Approved {approved}[/green] (all)"
        )
        return

    # Auto-threshold mode: approve if novelty >= threshold, else prompt
    for i, exp in enumerate(pending, 1):
        paper = get_paper(exp.paper_id)
        paper_title = paper.title if paper else exp.paper_id
        novelty = ""
        novelty_score = None
        if paper:
            analysis = get_analysis(exp.paper_id)
            if analysis:
                novelty_score = analysis.novelty_score
                novelty = f"  novelty={novelty_score:.1f}"

        # Auto-approve if novelty meets threshold
        if auto_threshold is not None and novelty_score is not None and novelty_score >= auto_threshold:
            update_experiment_status(exp.id, "pending")
            console.print(f"  [green]✓ Auto-approved[/green] '{exp.title}' (novelty={novelty_score:.1f} >= {auto_threshold})")
            approved += 1
            continue

        target_label = "~Modal GPU T4" if exp.execution_target == "cloud_modal" else "~local CPU"

        from rich.panel import Panel
        console.print(
            Panel(
                f"[bold]{exp.title}[/bold]\n"
                f"Paper: {paper_title}{novelty}\n"
                f"Hypothesis: {exp.hypothesis}\n"
                f"Target: {exp.execution_target}  |  Est. runtime: {target_label}",
                title=f"[{i}/{total}] Experiment Review",
                border_style="cyan",
            )
        )

        while True:
            choice = console.input(
                "  [A]pprove  [R]eject  [E]dit hypothesis  [S]kip all > "
            ).strip().upper()

            if choice == "A":
                update_experiment_status(exp.id, "pending")
                console.print("  [green]✓ Approved[/green]")
                approved += 1
                break
            elif choice == "R":
                update_experiment_status(exp.id, "skipped", error="Rejected by user")
                console.print("  [red]✗ Rejected[/red]")
                rejected += 1
                break
            elif choice == "E":
                console.print(f"  Current hypothesis: {exp.hypothesis}")
                new_hyp = console.input("  New hypothesis: ").strip()
                if new_hyp:
                    update_experiment_hypothesis(exp.id, new_hyp)
                update_experiment_status(exp.id, "pending")
                console.print("  [green]✓ Edited and approved[/green]")
                approved += 1
                break
            elif choice == "S":
                skipped_count = total - i
                console.print(f"  [yellow]Skipping remaining {skipped_count} experiments.[/yellow]")
                break
            else:
                console.print("  Please enter A, R, E, or S.")
        else:
            continue
        if choice == "S":
            break

    console.print(
        f"\n[bold]Review complete:[/bold] "
        f"[green]Approved {approved}[/green] / "
        f"[red]Rejected {rejected}[/red] / "
        f"[yellow]Skipped {skipped_count}[/yellow]"
    )


@app.command()
def scheduler():
    """Start the APScheduler daemon."""
    _configure_logging()
    from core.database import init_db, init_chroma
    init_db()
    init_chroma()

    console.print("[bold blue]Starting scheduler daemon...[/bold blue]")
    from scheduler.cron_scheduler import start
    start()


# Sub-app for experiment commands
experiment_app = typer.Typer()
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("run")
def experiment_run(
    exp_id: str = typer.Option(..., "--id", help="Experiment ID to run"),
):
    """Re-run a specific experiment by ID."""
    _configure_logging()
    from core.database import init_db
    from knowledge.experiment_store import get_experiment, save_result, delete_result, get_result, update_experiment_status
    from experiments import local_runner, cloud_runner, code_validator, router

    init_db()

    exp = get_experiment(exp_id)
    if exp is None:
        console.print(f"[red]Experiment {exp_id} not found[/red]")
        raise typer.Exit(1)

    # Validate
    validated_code, ok = code_validator.validate_with_retry(exp.generated_code, exp.paper_id)
    if not ok:
        console.print("[red]Code validation failed[/red]")
        raise typer.Exit(1)
    exp.generated_code = validated_code

    # Route
    target = router.decide_target(exp)
    console.print(f"Running experiment [cyan]{exp.title}[/cyan] on [yellow]{target}[/yellow]...")

    update_experiment_status(exp.id, "running")
    if get_result(exp.id):
        delete_result(exp.id)
    try:
        import concurrent.futures, time as _time
        from rich.live import Live
        from rich.spinner import Spinner
        from rich.text import Text

        runner_fn = local_runner.run if target == "local" else cloud_runner.run
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runner_fn, exp)
            start = _time.monotonic()
            with Live(refresh_per_second=4, console=console) as live:
                while not future.done():
                    elapsed = int(_time.monotonic() - start)
                    mins, secs = divmod(elapsed, 60)
                    live.update(Text(f"  Running on {target}... {mins:02d}:{secs:02d} elapsed", style="yellow"))
                    _time.sleep(0.25)
            result = future.result()

        save_result(result)
        no_metrics = result.metrics == "{}"
        if result.exit_code == 0 and not no_metrics:
            update_experiment_status(exp.id, "completed")
        else:
            error_msg = "no metrics produced" if (result.exit_code == 0 and no_metrics) else f"exit_code={result.exit_code}"
            update_experiment_status(exp.id, "failed", error=error_msg)
        console.print(f"[green]Done.[/green] Exit code: {result.exit_code}")
        if result.metrics and result.metrics != "{}":
            console.print(f"Metrics: {result.metrics}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        update_experiment_status(exp.id, "failed", error=str(e))
        raise typer.Exit(1)


@experiment_app.command("run-pending")
def experiment_run_pending():
    """Run all pending experiments (without a full pipeline cycle)."""
    _configure_logging()
    from core.database import init_db
    from core.state import new_state
    from datetime import datetime
    from experiments.experiment_pipeline import run as run_experiments

    init_db()

    from knowledge.experiment_store import get_experiments_by_status
    pending = get_experiments_by_status("pending")
    if not pending:
        console.print("[yellow]No pending experiments.[/yellow]")
        return

    console.print(f"[bold blue]Running {len(pending)} pending experiment(s)...[/bold blue]")
    state = new_state(f"manual_exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    run_experiments(state)

    from knowledge.experiment_store import get_experiments_by_status as get_by_status
    completed = len(get_by_status("completed"))
    failed = len(get_by_status("failed"))
    console.print(f"[green]Done.[/green] Check results with [cyan]uv run python main.py status[/cyan]")


@app.command()
def papers(
    status: str = typer.Option("", "--status", "-s", help="Filter by status: fetched|analyzed|done"),
    limit: int = typer.Option(30, "--limit", "-n"),
    search: str = typer.Option("", "--search", help="Filter by title keyword"),
):
    """List papers in the database."""
    _configure_logging()
    from knowledge.paper_store import get_all_papers, get_papers_by_status

    if status:
        paper_list = get_papers_by_status(status)
    else:
        paper_list = get_all_papers(limit=10000)

    if search:
        paper_list = [p for p in paper_list if search.lower() in p.title.lower()]

    paper_list = paper_list[:limit]

    if not paper_list:
        console.print("[yellow]No papers found.[/yellow]")
        return

    from rich.table import Table
    table = Table(title=f"Papers ({len(paper_list)} shown)", show_lines=False)
    table.add_column("ID", style="dim", no_wrap=True, max_width=28)
    table.add_column("Title", max_width=60)
    table.add_column("Source", max_width=8)
    table.add_column("Status", max_width=10)
    table.add_column("Published", max_width=12)

    for p in paper_list:
        pub = p.published_date.strftime("%Y-%m-%d") if p.published_date else ""
        table.add_row(p.id, p.title[:60], p.source, p.status, pub)

    console.print(table)
    console.print(f"\nUse [cyan]uv run python main.py synthesize --id <ID>[/cyan] to run synthesis on a specific paper.")


@app.command()
def synthesize(
    paper_id: str = typer.Option(..., "--id", help="Paper ID to synthesize"),
):
    """Run experiment generation on a specific paper (analyzes first if not yet analyzed)."""
    _configure_logging()
    from core.database import init_db, init_chroma

    init_db()
    init_chroma()

    from knowledge.paper_store import get_paper, get_analysis, save_analysis, update_paper_status, update_paper_full_text
    from knowledge.experiment_store import save_experiment
    from synthesis import paper_analyzer, experiment_extractor
    from knowledge import vector_store

    paper = get_paper(paper_id)
    if paper is None:
        console.print(f"[red]Paper '{paper_id}' not found in database.[/red]")
        console.print("Run [cyan]uv run python main.py papers[/cyan] to list available papers.")
        raise typer.Exit(1)

    console.print(f"\n[bold]Paper:[/bold] {paper.title}")
    console.print(f"[dim]Source:[/dim] {paper.source}  |  [dim]Status:[/dim] {paper.status}")

    # Use existing analysis if available, otherwise run Claude
    analysis = get_analysis(paper_id)
    if analysis:
        console.print(f"  Using existing analysis — novelty={analysis.novelty_score:.1f} relevance={analysis.relevance_score:.1f}")
    else:
        # Fetch full text for arXiv papers
        if paper.source == "arxiv" and not paper.full_text:
            console.print("  Fetching full text from arXiv...")
            from ingestion.fulltext_fetcher import fetch_arxiv_fulltext
            ft = fetch_arxiv_fulltext(paper.source_id)
            if ft:
                paper.full_text = ft
                update_paper_full_text(paper.id, ft)
                console.print(f"  [green]✓[/green] Fetched {len(ft):,} chars")

        console.print("  Analyzing paper with Claude...")
        analysis = paper_analyzer.analyze_paper(paper)
        save_analysis(analysis)
        update_paper_status(paper_id, "analyzed")
        update_paper_full_text(paper_id, None)
        console.print(
            f"  [green]✓[/green] Analysis done — novelty={analysis.novelty_score:.1f} "
            f"relevance={analysis.relevance_score:.1f} difficulty={analysis.reproducibility_difficulty}"
        )

    # Contradiction check
    try:
        from knowledge import contradiction_detector
        contradiction_detector.check_new_paper(paper_id, analysis)
    except Exception as e:
        log.warning("synthesize.contradiction_check_failed", error=str(e))

    # Embed
    vector_store.embed_paper(paper)
    console.print("  [green]✓[/green] Embedded into vector store")

    # Extract experiments (note: extractor skips paper if experiments already exist)
    from config import settings
    if analysis.relevance_score < settings.min_relevance_score_to_experiment:
        console.print(
            f"\n[yellow]Relevance score {analysis.relevance_score:.1f} is below threshold "
            f"{settings.min_relevance_score_to_experiment} — skipping experiment generation.[/yellow]"
        )
        console.print("Use [cyan]--force[/cyan] or lower [cyan]min_relevance_score_to_experiment[/cyan] in .env to override.")
        return

    console.print("  Extracting experiments...")
    experiments = experiment_extractor.extract_experiments(paper_id, analysis)
    if not experiments:
        console.print("  [yellow]No experiments generated[/yellow] (paper may already have experiments — check `review`)")
    else:
        for exp in experiments:
            save_experiment(exp)
        console.print(f"  [green]✓[/green] {len(experiments)} experiment(s) created → [bold]pending_review[/bold]")
        console.print(f"\nRun [cyan]uv run python main.py review[/cyan] to approve and queue them for execution.")


@app.command()
def watch(
    refresh: int = typer.Option(5, "--refresh", "-r", help="Refresh interval in seconds"),
):
    """Live dashboard showing pipeline state, refreshing every N seconds."""
    _configure_logging()
    from config import settings
    from core.state import find_incomplete_states
    from knowledge.experiment_store import get_experiments_by_status
    from knowledge.paper_store import get_all_papers
    import datetime as dt

    def _build_table() -> Table:
        table = Table(title="Research Agent — Live Status", expand=True)
        table.add_column("Item", style="bold cyan")
        table.add_column("Value")

        incomplete = find_incomplete_states()
        if incomplete:
            s = incomplete[-1]
            table.add_row("Active Cycle", s.cycle_id)
            table.add_row("Current Stage", f"[blue]{s.current_stage}[/blue]")
            table.add_row("Completed Stages", ", ".join(s.completed_stages) or "none")
            table.add_row("Papers This Cycle", str(len(s.paper_ids_this_cycle)))
        else:
            table.add_row("Pipeline", "[green]Idle[/green]")
        table.add_section()

        colors = {"pending_review": "magenta", "pending": "yellow", "running": "blue",
                  "completed": "green", "failed": "red", "skipped": "dim"}
        for status_name in ["pending_review", "pending", "running", "completed", "failed", "skipped"]:
            c = colors[status_name]
            n = len(get_experiments_by_status(status_name))
            table.add_row(f"Experiments [{status_name}]", f"[{c}]{n}[/{c}]")
        table.add_section()

        table.add_row("Total Papers in DB", str(len(get_all_papers(limit=10000))))
        state_files = sorted(settings.state_dir.glob("*.json"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if state_files:
            mtime = state_files[0].stat().st_mtime
            table.add_row("Last State Write", dt.datetime.fromtimestamp(mtime).strftime("%H:%M:%S"))
        table.add_section()

        from core import token_tracker
        totals = token_tracker.get_totals()
        inp = totals["input_total"]
        out = totals["output_total"]
        cost = inp * 3 / 1_000_000 + out * 15 / 1_000_000
        table.add_row("Tokens this cycle", f"in={inp:,} out={out:,}")
        table.add_row("Est. cost", f"${cost:.4f}")
        return table

    console.print("[bold blue]Watching pipeline — Ctrl+C to stop[/bold blue]")
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                live.update(_build_table())
                time.sleep(refresh)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


if __name__ == "__main__":
    app()
