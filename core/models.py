"""SQLModel + Pydantic v2 data models for the research agent."""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, field_validator
from sqlmodel import Field, SQLModel


# ---------------------------------------------------------------------------
# SQLite-backed models (table=True)
# ---------------------------------------------------------------------------

class Paper(SQLModel, table=True):
    __tablename__ = "paper"

    id: str = Field(primary_key=True)          # SHA256(source_id)
    title: str
    abstract: str
    source: str                                 # "arxiv" | "semantic_scholar" | "substack"
    source_id: str
    url: str
    pdf_url: Optional[str] = None
    published_date: date
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    full_text: Optional[str] = None
    tags: str = "[]"                            # JSON list
    citation_count: Optional[int] = None
    status: str = Field(default="fetched", index=True)  # "fetched" | "analyzed" | "done"


class PaperAnalysis(SQLModel, table=True):
    __tablename__ = "paper_analysis"

    id: str = Field(primary_key=True)
    paper_id: str = Field(index=True)
    key_contributions: str = "[]"              # JSON list[str]
    methods_described: str = "[]"              # JSON list[str]
    reproducible_experiments: str = "[]"       # JSON list[dict]
    novelty_score: float = 0.0                 # 1-10
    relevance_score: float = 0.0              # 1-10
    limitations: str = "[]"                   # JSON list[str] — stated limitations
    datasets_used: str = "[]"                 # JSON list[str] — datasets mentioned
    key_hyperparameters: str = "{}"           # JSON dict — important hyperparameters
    reproducibility_difficulty: str = "medium"  # "easy" | "medium" | "hard"
    raw_claude_response: str = ""
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class Experiment(SQLModel, table=True):
    __tablename__ = "experiment"

    id: str = Field(primary_key=True)          # UUID
    paper_id: str = Field(index=True)
    title: str
    hypothesis: str
    generated_code: str = ""
    execution_target: str = "local"            # "local" | "cloud_modal"
    status: str = Field(default="pending_review", index=True)  # "pending_review"|"pending"|"running"|"completed"|"failed"|"skipped"
    parent_experiment_id: Optional[str] = None # set for ablation variants
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class ExperimentResult(SQLModel, table=True):
    __tablename__ = "experiment_result"

    id: str = Field(primary_key=True)
    experiment_id: str = Field(index=True)
    stdout: str = ""
    exit_code: int = 0
    metrics: str = "{}"                        # JSON dict
    artifacts: str = "[]"                      # JSON list of file paths
    runtime_seconds: float = 0.0
    statistical_summary: Optional[str] = None  # JSON
    baseline_comparison: Optional[str] = None  # JSON
    conclusion: Optional[str] = None           # Claude-generated
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


class ResearchReport(SQLModel, table=True):
    __tablename__ = "research_report"

    id: str = Field(primary_key=True)
    cycle_id: str = Field(index=True)
    title: str
    report_type: str = "weekly"               # "weekly" | "experiment_deep_dive"
    paper_ids: str = "[]"                      # JSON list
    experiment_ids: str = "[]"                 # JSON list
    markdown_content: str = ""
    key_findings: str = "[]"                   # JSON list
    open_questions: str = "[]"                 # JSON list
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class Contradiction(SQLModel, table=True):
    __tablename__ = "contradiction"

    id: str = Field(primary_key=True)
    paper_id_new: str = Field(index=True)      # newer paper making the claim
    paper_id_old: str = Field(index=True)      # older paper being contradicted
    metric: str
    description: str
    severity: str = "partial"                  # "direct" | "partial" | "methodological"
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class ResearchGap(SQLModel, table=True):
    __tablename__ = "research_gap"

    id: str = Field(primary_key=True)
    description: str
    supporting_paper_ids: str = "[]"           # JSON list[str]
    cycle_id: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class ThemeCluster(SQLModel, table=True):
    __tablename__ = "theme_cluster"

    id: str = Field(primary_key=True)
    name: str
    description: str
    paper_ids: str = "[]"                      # JSON list[str]
    paper_count: int = 0
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TokenUsageLog(SQLModel, table=True):
    __tablename__ = "token_usage_log"

    id: str = Field(primary_key=True)          # UUID
    cycle_id: str = Field(index=True)
    module: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


class EvalMetric(SQLModel, table=True):
    __tablename__ = "eval_metric"

    id: str = Field(primary_key=True)            # uuid4
    metric: str = Field(index=True)              # "reproduction_rate" | "partial_rate" | (future)
    dimension: str = Field(index=True, default="overall")  # "overall" | "difficulty:easy" | "target:local" | ...
    value: Optional[float] = None                # fraction 0-1; None if denominator == 0
    numerator: int = 0
    denominator: int = 0
    cycle_id: str = Field(index=True)            # real cycle_id, or "backfill-<YYYY-Www>"
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    context: str = "{}"                          # JSON: {"fully":3,"partial":1,"not":2}


class BenchmarkItem(SQLModel, table=True):
    __tablename__ = "benchmark_item"

    id: str = Field(primary_key=True)                 # uuid4
    experiment_id: str = Field(index=True)            # experiment this claim is scored against
    metric_name: str                                  # key in ExperimentResult.metrics JSON
    expected_value: float                             # the paper's known/true value
    tolerance: float                                  # band half-width
    tolerance_type: str = "relative"                  # "relative" (% of expected) | "absolute"
    unit: Optional[str] = None                        # display only
    note: str = ""                                    # provenance / why this is ground truth
    active: bool = True                               # deactivate instead of hard-delete
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkRun(SQLModel, table=True):
    __tablename__ = "benchmark_run"

    id: str = Field(primary_key=True)                 # uuid4
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    cycle_id: Optional[str] = Field(default=None, index=True)  # pipeline cycle, or None for manual
    trigger: str = "manual"                            # "cycle" | "manual"
    n_items: int = 0
    n_pass: int = 0
    n_fail: int = 0
    n_unscorable: int = 0
    accuracy: Optional[float] = None                   # n_pass / scorable; None if 0 scorable


class BenchmarkItemResult(SQLModel, table=True):
    __tablename__ = "benchmark_item_result"

    id: str = Field(primary_key=True)                 # uuid4
    run_id: str = Field(index=True)
    item_id: str = Field(index=True)
    experiment_id: str
    metric_name: str
    expected_value: float
    tolerance: float
    tolerance_type: str
    measured_value: Optional[float] = None            # None when unscorable
    passed: Optional[bool] = None                     # None when unscorable
    status: str                                        # pass|fail|no_result|missing_metric|non_numeric


# ---------------------------------------------------------------------------
# In-memory / file-backed models (Pydantic only)
# ---------------------------------------------------------------------------

class RunState(BaseModel):
    """Crash-safe pipeline run state, serialized atomically to JSON."""
    cycle_id: str
    started_at: datetime
    current_stage: str = "ingestion"           # pipeline stage name
    completed_stages: list[str] = []
    paper_ids_this_cycle: list[str] = []
    experiment_ids_this_cycle: list[str] = []
    error_log: list[dict] = []
    is_complete: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0


# ---------------------------------------------------------------------------
# Typed helpers for JSON column deserialization
# ---------------------------------------------------------------------------

def parse_json_list(value: str) -> list:
    try:
        return json.loads(value)
    except Exception:
        return []


def parse_json_dict(value: str) -> dict:
    try:
        return json.loads(value)
    except Exception:
        return {}
