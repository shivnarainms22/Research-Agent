"""Central configuration via pydantic-settings (reads from .env)."""
from __future__ import annotations

from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    anthropic_api_key: str = ""
    semantic_scholar_api_key: str = ""
    modal_token_id: str = ""
    modal_token_secret: str = ""

    # Claude models
    claude_model: str = "claude-sonnet-4-6"
    claude_haiku_model: str = "claude-haiku-4-5-20251001"

    # Paths
    base_dir: Path = Path("D:/Research")
    data_dir: Path = Path("D:/Research/data")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "research.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "db" / "chroma"

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    # ArXiv
    arxiv_categories: list[str] = ["cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.RO"]
    arxiv_keywords: list[str] = [
        # Mechanistic interpretability core
        "mechanistic interpretability",
        "sparse autoencoder",
        "superposition",
        "polysemanticity",
        "feature visualization",
        "transformer circuits",
        "activation patching",
        "steering vector",
        "representation engineering",
        # Probing / linear representation
        "probing classifier",
        "linear representation",
        "concept bottleneck",
        # VLA / embodied
        "vision-language-action",
        "VLA model",
        "robot learning",
    ]
    max_papers_per_cycle: int = 10

    # Substack RSS feeds
    substack_rss_feeds: list[str] = [
        "https://transformer-circuits.pub/feed.xml",
    ]

    # Synthesis thresholds
    min_novelty_score_to_experiment: float = 7.5
    min_relevance_score_to_experiment: float = 7.0
    min_keyword_matches_to_analyze: int = 2

    # Experiment execution
    experiment_timeout_seconds: int = 3600
    docker_memory_limit: str = "8g"
    docker_cpu_limit: float = 4.0
    enable_experiment_network: bool = True

    # Scheduler
    ingestion_hours: list[int] = [6, 18]
    experiment_poll_minutes: int = 30

    # Logging
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _load_domain_yaml(self) -> "Settings":
        """Override settings from domain.yaml if the file exists."""
        domain_yaml = Path("D:/Research/domain.yaml")
        if not domain_yaml.exists():
            return self
        try:
            import yaml
            data = yaml.safe_load(domain_yaml.read_text(encoding="utf-8")) or {}
            if "arxiv_categories" in data:
                object.__setattr__(self, "arxiv_categories", data["arxiv_categories"])
            if "keywords" in data:
                object.__setattr__(self, "arxiv_keywords", data["keywords"])
            thresholds = data.get("thresholds", {})
            if "min_novelty_score_to_experiment" in thresholds:
                object.__setattr__(
                    self,
                    "min_novelty_score_to_experiment",
                    float(thresholds["min_novelty_score_to_experiment"]),
                )
            if "min_relevance_score_to_experiment" in thresholds:
                object.__setattr__(
                    self,
                    "min_relevance_score_to_experiment",
                    float(thresholds["min_relevance_score_to_experiment"]),
                )
            if "min_keyword_matches_to_analyze" in thresholds:
                object.__setattr__(
                    self,
                    "min_keyword_matches_to_analyze",
                    int(thresholds["min_keyword_matches_to_analyze"]),
                )
        except Exception:
            pass  # domain.yaml loading is best-effort
        return self


settings = Settings()


def ensure_dirs() -> None:
    """Create all required data directories."""
    for d in [
        settings.data_dir / "db",
        settings.chroma_path,
        settings.papers_dir,
        settings.experiments_dir,
        settings.reports_dir,
        settings.state_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)
