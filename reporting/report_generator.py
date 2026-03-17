"""Claude-powered research report generator with Jinja2 templates."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import anthropic
import jinja2
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from core import token_tracker
from core.models import ResearchReport, RunState, parse_json_list
from knowledge.contradiction_store import get_recent_contradictions
from knowledge.experiment_store import get_experiments_by_status, get_result, get_ablations_for_parent, get_all_experiments
from knowledge.gap_store import get_gaps
from knowledge.paper_store import get_paper, get_analysis, get_all_papers, get_papers_by_status
from knowledge.theme_store import get_all_themes

log = structlog.get_logger()

_NOISE_KEYS = {"n", "SD", "se", "df", "p", "t", "z", "f"}


def _summarize_metrics(metrics: dict) -> str:
    """Convert a raw metrics dict to a human-readable bullet list."""
    lines = []
    for k, v in metrics.items():
        if len(str(k)) <= 1:
            continue
        if str(k).isdigit():
            continue
        if k in _NOISE_KEYS:
            continue
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) if lines else "- (no numeric results recorded)"


def _baseline_label(status: str) -> str:
    """Map raw baseline_status to a plain-English label."""
    return {
        "fully_reproduced": "✓ Matched paper claims",
        "partially_reproduced": "~ Partially matched paper claims",
        "not_reproduced": "✗ Did not match paper claims",
    }.get(status, "? No baseline available")


_TEMPLATE_DIR = Path(__file__).parent / "templates"

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
def _generate_narrative(
    papers_summary: str,
    experiments_summary: str,
    cycle_id: str,
) -> dict:
    """Ask Claude to synthesize findings into narrative sections."""
    client = _get_client()

    system = """\
You are a researcher writing a weekly research digest.
Write in clear, accessible language that a smart person outside your field could understand. Avoid jargon.
Say 'the model learned to ignore irrelevant features' not 'the model exhibited suppression of non-salient representational components.'
Focus on: interpretability, computer vision, and vision-language-action models.
"""

    tool = {
        "name": "write_report",
        "description": "Write structured research report sections",
        "input_schema": {
            "type": "object",
            "properties": {
                "tldr": {"type": "string", "description": "1-2 sentence plain-English gist. Write as if texting a smart friend who doesn't read papers."},
                "executive_summary": {"type": "string", "description": "3-5 sentence overview, plain-English, no unexplained acronyms"},
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "5-10 bullet points of key findings; each should be one clear sentence a non-expert can parse",
                },
                "open_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 open research questions raised this cycle, written as plain-English questions",
                },
                "next_experiments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 concrete follow-up experiments; state concretely what would be measured and why",
                },
            },
            "required": ["tldr", "executive_summary", "key_findings", "open_questions", "next_experiments"],
        },
    }

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=4096,
        temperature=0.7,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": f"""Research cycle {cycle_id} summary:

## Papers Analyzed
{papers_summary[:4000]}

## Experiments Results
{experiments_summary[:2000]}

Please synthesize these into report sections."""
        }],
        tools=[tool],
        tool_choice={"type": "tool", "name": "write_report"},
    )

    log.info(
        "claude.usage",
        module="report_generator",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
    )
    token_tracker.track("report_generator", response.usage.input_tokens, response.usage.output_tokens)

    result = next(
        (b.input for b in response.content if b.type == "tool_use"),
        None,
    )
    return result or {}


def generate(state: RunState, report_type: str = "weekly") -> ResearchReport:
    """Generate a full research report for this cycle."""
    # Always report on all analyzed papers and all experiments in the DB —
    # reports are comprehensive snapshots, not single-cycle views.
    paper_ids = [p.id for p in get_papers_by_status("analyzed")]
    log.info("report_generator.papers_scope", count=len(paper_ids))

    exp_ids_in_scope = {e.id for e in get_all_experiments()}
    log.info("report_generator.experiments_scope", count=len(exp_ids_in_scope))

    _paper_title_lookup = {p.id: p.title for p in get_all_papers(limit=10000)}

    # Gather paper data
    paper_sections = []
    papers_text_parts = []
    for paper_id in paper_ids[:20]:
        paper = get_paper(paper_id)
        analysis = get_analysis(paper_id)
        if paper is None:
            continue

        novelty = analysis.novelty_score if analysis else 0.0
        if novelty == 0.0:
            continue

        section = {
            "title": paper.title,
            "source": paper.source,
            "published_date": str(paper.published_date),
            "url": paper.url,
            "novelty_score": novelty,
            "relevance_score": analysis.relevance_score if analysis else 0,
            "key_contributions": parse_json_list(analysis.key_contributions) if analysis else [],
            "limitations": parse_json_list(analysis.limitations) if analysis else [],
            "datasets_used": parse_json_list(analysis.datasets_used) if analysis else [],
            "key_hyperparameters": json.loads(analysis.key_hyperparameters) if analysis and analysis.key_hyperparameters else {},
            "reproducibility_difficulty": analysis.reproducibility_difficulty if analysis else "unknown",
        }
        paper_sections.append(section)
        papers_text_parts.append(
            f"- {paper.title} (novelty={section['novelty_score']:.1f}, difficulty={section['reproducibility_difficulty']}): "
            + "; ".join(section["key_contributions"][:2])
            + (f" | datasets: {', '.join(section['datasets_used'][:2])}" if section["datasets_used"] else "")
            + (f" | limitations: {section['limitations'][0]}" if section["limitations"] else "")
        )

    # Gather experiment data
    exp_sections = []
    exp_text_parts = []
    all_completed = get_experiments_by_status("completed")
    cycle_exp_ids = exp_ids_in_scope

    for exp in all_completed:
        if exp.id not in cycle_exp_ids:
            continue
        result = get_result(exp.id)
        metrics = json.loads(result.metrics) if result and result.metrics else {}
        comparison = json.loads(result.baseline_comparison) if result and result.baseline_comparison else {}

        section = {
            "id": exp.id,
            "title": exp.title,
            "paper_id": exp.paper_id,
            "paper_title": _paper_title_lookup.get(exp.paper_id, exp.paper_id),
            "status": exp.status,
            "execution_target": exp.execution_target,
            "runtime_seconds": result.runtime_seconds if result else 0,
            "hypothesis": exp.hypothesis,
            "metrics_bullets": _summarize_metrics(metrics),
            "conclusion": result.conclusion if result and result.conclusion else "",
            "baseline_status": comparison.get("overall", "unknown"),
            "baseline_label": _baseline_label(comparison.get("overall", "unknown")),
            "exit_code": result.exit_code if result else -1,
            "parent_experiment_id": exp.parent_experiment_id,
        }
        exp_sections.append(section)
        exp_text_parts.append(
            f"- {exp.title}: status={exp.status}, baseline={section['baseline_status']}"
        )

    # Build ablation families: group parent experiments with their ablation variants
    parent_sections = [s for s in exp_sections if s.get("parent_experiment_id") is None]
    ablation_map: dict[str, list] = {}
    for exp in all_completed:
        if exp.id not in cycle_exp_ids or exp.parent_experiment_id is None:
            continue
        abl_result = get_result(exp.id)
        abl_metrics = json.loads(abl_result.metrics) if abl_result and abl_result.metrics else {}
        abl_comparison = json.loads(abl_result.baseline_comparison) if abl_result and abl_result.baseline_comparison else {}
        abl_section = {
            "id": exp.id,
            "title": exp.title,
            "paper_id": exp.paper_id,
            "paper_title": _paper_title_lookup.get(exp.paper_id, exp.paper_id),
            "status": exp.status,
            "execution_target": exp.execution_target,
            "runtime_seconds": abl_result.runtime_seconds if abl_result else 0,
            "hypothesis": exp.hypothesis,
            "metrics_bullets": _summarize_metrics(abl_metrics),
            "conclusion": abl_result.conclusion if abl_result and abl_result.conclusion else "",
            "baseline_status": abl_comparison.get("overall", "unknown"),
            "baseline_label": _baseline_label(abl_comparison.get("overall", "unknown")),
            "exit_code": abl_result.exit_code if abl_result else -1,
            "parent_experiment_id": exp.parent_experiment_id,
        }
        ablation_map.setdefault(exp.parent_experiment_id, []).append(abl_section)

    # Also fetch ablations that were created in previous cycles but linked to cycle parents
    for parent_sec in parent_sections:
        if parent_sec["id"] not in ablation_map:
            db_ablations = get_ablations_for_parent(parent_sec["id"])
            for abl_exp in db_ablations:
                if abl_exp.id in cycle_exp_ids:
                    continue  # already captured above
                abl_result = get_result(abl_exp.id)
                abl_metrics = json.loads(abl_result.metrics) if abl_result and abl_result.metrics else {}
                abl_comparison = json.loads(abl_result.baseline_comparison) if abl_result and abl_result.baseline_comparison else {}
                abl_section = {
                    "id": abl_exp.id,
                    "title": abl_exp.title,
                    "paper_id": abl_exp.paper_id,
                    "paper_title": _paper_title_lookup.get(abl_exp.paper_id, abl_exp.paper_id),
                    "status": abl_exp.status,
                    "execution_target": abl_exp.execution_target,
                    "runtime_seconds": abl_result.runtime_seconds if abl_result else 0,
                    "hypothesis": abl_exp.hypothesis,
                    "metrics_bullets": _summarize_metrics(abl_metrics),
                    "conclusion": abl_result.conclusion if abl_result and abl_result.conclusion else "",
                    "baseline_status": abl_comparison.get("overall", "unknown"),
                    "baseline_label": _baseline_label(abl_comparison.get("overall", "unknown")),
                    "exit_code": abl_result.exit_code if abl_result else -1,
                    "parent_experiment_id": abl_exp.parent_experiment_id,
                }
                ablation_map.setdefault(abl_exp.parent_experiment_id, []).append(abl_section)

    experiment_families = [
        {"parent": s, "ablations": ablation_map.get(s["id"], [])}
        for s in parent_sections
    ]

    # Generate narrative (skip if nothing to report)
    if not papers_text_parts and not exp_text_parts:
        log.info("report_generator.narrative_skipped", reason="no_content")
        narrative = {}
    else:
        narrative = _generate_narrative(
            papers_summary="\n".join(papers_text_parts),
            experiments_summary="\n".join(exp_text_parts),
            cycle_id=state.cycle_id,
        )

    next_experiments = narrative.get("next_experiments", [])

    # Fetch living review data
    contradictions = get_recent_contradictions(days=30)
    gaps = get_gaps()
    themes = get_all_themes()

    # Count stats
    total_papers = len(get_all_papers(limit=10000))
    total_experiments = len(all_completed)
    reproduced = sum(
        1 for s in exp_sections
        if s["baseline_status"] in ("fully_reproduced", "partially_reproduced")
    )
    reproduction_rate = round(reproduced / max(len(exp_sections), 1) * 100, 1)

    # Render Markdown via Jinja2
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,
    )
    template = env.get_template("weekly_report.md.j2")
    md = template.render(
        title=f"Weekly Research Digest — Cycle {state.cycle_id}",
        cycle_id=state.cycle_id,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        report_type=report_type,
        tldr=narrative.get("tldr", ""),
        executive_summary=narrative.get("executive_summary", ""),
        papers=paper_sections,
        experiments=exp_sections,
        experiment_families=experiment_families,
        key_findings=narrative.get("key_findings", []),
        open_questions=narrative.get("open_questions", []),
        next_experiments=next_experiments,
        total_papers=total_papers,
        total_experiments=total_experiments,
        reproduction_rate=reproduction_rate,
        contradictions=contradictions,
        gaps=gaps,
        themes=themes,
    )

    # Save Markdown file
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.reports_dir / f"{state.cycle_id}_{report_type}.md"
    report_path.write_text(md, encoding="utf-8")
    log.info("report_generator.saved", path=str(report_path))
    log.info("report_generator.paper_filter", included=len(paper_sections))

    report_id = str(uuid.uuid4())
    report = ResearchReport(
        id=report_id,
        cycle_id=state.cycle_id,
        title=f"Weekly Research Digest — Cycle {state.cycle_id}",
        report_type=report_type,
        paper_ids=json.dumps(state.paper_ids_this_cycle),
        experiment_ids=json.dumps(state.experiment_ids_this_cycle),
        markdown_content=md,
        key_findings=json.dumps(narrative.get("key_findings", [])),
        open_questions=json.dumps(narrative.get("open_questions", [])),
        generated_at=datetime.utcnow(),
    )

    # Save to SQLite
    from core.database import get_engine
    from sqlmodel import Session
    with Session(get_engine(), expire_on_commit=False) as session:
        session.add(report)
        session.commit()

    return report
