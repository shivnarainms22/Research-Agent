# Research Agent

An autonomous research system that continuously discovers papers, analyzes them with AI, generates and runs experiments, and produces weekly reports — closing the loop from literature review to empirical results.

## What It Does

```
arxiv / Semantic Scholar / Substack
        │
        ▼
   ┌─────────┐     ┌───────────┐     ┌─────────────┐
   │ Ingest  │────▶│ Synthesize│────▶│  Experiment  │
   │ papers  │     │ & analyze │     │  (Docker/GPU)│
   └─────────┘     └───────────┘     └──────┬───────┘
                                            │
        ┌───────────┐     ┌─────────┐       │
        │  Report   │◀────│ Analyze │◀──────┘
        │ (Markdown)│     │ results │
        └───────────┘     └─────────┘
```

**Pipeline stages:**

1. **Ingestion** — Fetches papers from arXiv, Semantic Scholar, and Substack RSS feeds. Deduplicates across sources (exact + semantic).
2. **Synthesis** — Claude analyzes each paper (novelty, relevance, reproducibility), generates embeddings, detects contradictions against existing literature, and extracts experiment candidates.
3. **Experiments** — Validated code (AST + Bandit security scan) runs in Docker sandboxes (CPU) or Modal cloud GPUs (T4). All experiments require human approval before execution.
4. **Analysis** — Statistical analysis (CI, t-tests, Cohen's d), baseline comparison against paper-claimed metrics, and ablation study generation.
5. **Reporting** — Claude generates narrative Markdown reports with Jinja2 templates covering findings, contradictions, research gaps, and theme clusters.

The system also maintains a **knowledge graph** (NetworkX), **vector store** (ChromaDB), and **hybrid retrieval** (BM25 + vector RRF) for cross-paper reasoning.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker (for local experiment sandboxing)
- [Modal](https://modal.com/) account (optional, for GPU experiments)

### Installation

```bash
git clone https://github.com/<your-username>/research-agent.git
cd research-agent

# Install dependencies
uv sync

# Copy and fill in your API keys
cp .env.example .env

# Initialize database, ChromaDB, and validate API keys
uv run python main.py setup
```

### Environment Variables

See [`.env.example`](.env.example) for all available options. Only `ANTHROPIC_API_KEY` is required.

## Usage

### CLI Commands

```bash
# Run a full pipeline cycle (ingest → synthesize → experiment → analyze → report)
uv run python main.py run

# Ingest papers from the last N days
uv run python main.py ingest --source arxiv --days 7

# Start the scheduler daemon (automated cycles)
uv run python main.py scheduler

# Check pipeline status
uv run python main.py status

# Live dashboard (token usage, pipeline state)
uv run python main.py watch

# View token usage, costs, and pipeline stats
uv run python main.py metrics

# Generate a report from the latest cycle
uv run python main.py report

# Review pending experiments (approve/reject/edit)
uv run python main.py review
uv run python main.py review --approve-all
uv run python main.py review --auto-threshold 7.5

# Re-run a specific experiment
uv run python main.py experiment run --id <uuid>
```

### Web UI

```bash
uv run streamlit run ui/app.py
```

Opens at `http://localhost:8501` with pages for:
- Dashboard overview
- Paper browser
- Experiment review queue
- Living review (contradictions, gaps, themes)
- Report viewer
- Manual paper ingestion

### Scheduler

The daemon runs on a configurable schedule (default):

| Job | Schedule |
|-----|----------|
| Paper ingestion | 6:00 AM, 6:00 PM |
| Experiment polling | Every 30 min |
| Gap analysis | Sunday 9:00 PM |
| Weekly report | Sunday 10:00 PM |
| Knowledge graph rebuild | Wednesday 3:00 AM |
| Theme clustering | Wednesday 3:30 AM |

## Domain Configuration

Edit `domain.yaml` to customize the research focus without touching code:

```yaml
arxiv_categories: [cs.LG, cs.AI, cs.CL, cs.CV, cs.RO]
keywords:
  - mechanistic interpretability
  - sparse autoencoder
  - superposition
  # ... add your own
thresholds:
  min_novelty_score_to_experiment: 7.5
  min_relevance_score_to_experiment: 7.0
  min_keyword_matches_to_analyze: 2
```

## Experiment Safety

All generated experiment code goes through multiple safety gates:

- **AST validation** — blocks `subprocess`, `eval`, `exec`, `__import__`, `os.system`, `shutil.rmtree`, etc.
- **Bandit scan** — rejects HIGH severity findings
- **Auto-fix** — Claude attempts to fix validation failures before giving up
- **Human approval** — all experiments start as `pending_review` and must be explicitly approved
- **Docker isolation** — sandboxed with `--memory 8g`, `--cpus 4`, `user=nobody`
- **GPU routing** — CPU-only tasks run locally; GPU tasks route to Modal (T4)

## Project Structure

```
research-agent/
├── main.py                  # Typer CLI entry point
├── config.py                # Pydantic settings + domain.yaml loader
├── domain.yaml              # Research domain configuration
├── core/
│   ├── models.py            # SQLModel tables (Paper, Experiment, etc.)
│   ├── database.py          # SQLite (WAL) + ChromaDB initialization
│   ├── state.py             # Atomic pipeline state management
│   └── token_tracker.py     # Thread-safe API token usage tracking
├── ingestion/
│   ├── ingestion_pipeline.py
│   ├── arxiv_client.py      # arXiv API
│   ├── semantic_scholar_client.py
│   ├── substack_scraper.py  # RSS + BeautifulSoup
│   ├── fulltext_fetcher.py  # arXiv HTML full-text
│   └── deduplicator.py      # Exact + semantic deduplication
├── synthesis/
│   ├── synthesis_pipeline.py  # Two-phase parallel architecture
│   ├── paper_analyzer.py      # Claude tool-use analysis
│   ├── experiment_extractor.py # Claude code generation
│   └── knowledge_graph.py     # NetworkX graph builder
├── experiments/
│   ├── experiment_pipeline.py
│   ├── code_validator.py    # AST + Bandit checks
│   ├── router.py            # Local vs cloud routing
│   ├── local_runner.py      # Docker sandbox
│   ├── cloud_runner.py      # Modal GPU (T4)
│   └── result_collector.py  # Metrics parsing
├── analysis/
│   ├── analysis_pipeline.py
│   ├── statistical_analyzer.py  # scipy stats
│   ├── baseline_comparator.py   # vs paper claims
│   └── ablation_manager.py      # Ablation study generation
├── knowledge/
│   ├── paper_store.py       # Paper CRUD
│   ├── experiment_store.py  # Experiment CRUD
│   ├── vector_store.py      # ChromaDB operations
│   ├── retriever.py         # Hybrid BM25 + vector search
│   ├── contradiction_detector.py
│   ├── gap_finder.py
│   └── theme_clusterer.py
├── reporting/
│   ├── report_generator.py  # Claude narrative + Jinja2
│   └── templates/
├── scheduler/
│   ├── pipeline_runner.py   # Orchestrator + crash recovery
│   └── cron_scheduler.py    # APScheduler jobs
├── ui/
│   └── app.py               # Streamlit web interface
├── docker/
│   ├── Dockerfile.sandbox
│   └── docker-compose.yml
└── tests/                   # 39 tests across 8 test files
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| AI | Claude API (Sonnet) via tool-use |
| Database | SQLite (WAL mode) + SQLModel |
| Vector store | ChromaDB + sentence-transformers |
| Retrieval | Hybrid BM25 + vector (RRF fusion) |
| Scheduling | APScheduler |
| Experiment sandbox | Docker + Modal (cloud GPU) |
| CLI | Typer + Rich |
| Web UI | Streamlit |
| Reports | Jinja2 + Markdown |
| Knowledge graph | NetworkX |

## Testing

```bash
uv run pytest tests/ -v --tb=short
```

Covers: deduplication, statistical analysis, baseline comparison, result collection, code validation, token tracking, synthesis pipeline, and experiment pipeline.

## License

All rights reserved.
