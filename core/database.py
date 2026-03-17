"""SQLite + ChromaDB initialization."""
from __future__ import annotations

import structlog
from sqlalchemy import text
from sqlmodel import SQLModel, create_engine, Session

from config import settings, ensure_dirs

log = structlog.get_logger()

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        ensure_dirs()
        db_url = f"sqlite:///{settings.db_path}?check_same_thread=False"
        _engine = create_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for better concurrent read performance
        with _engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            conn.commit()
    return _engine


def _migrate_db(engine) -> None:
    """Add new columns to existing tables without losing data."""
    new_columns = [
        "ALTER TABLE paper_analysis ADD COLUMN limitations TEXT DEFAULT '[]'",
        "ALTER TABLE paper_analysis ADD COLUMN datasets_used TEXT DEFAULT '[]'",
        "ALTER TABLE paper_analysis ADD COLUMN key_hyperparameters TEXT DEFAULT '{}'",
        "ALTER TABLE paper_analysis ADD COLUMN reproducibility_difficulty TEXT DEFAULT 'medium'",
    ]
    with engine.connect() as conn:
        for stmt in new_columns:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass  # column already exists


def init_db() -> None:
    """Create all tables if they don't exist."""
    from core.models import Paper, PaperAnalysis, Experiment, ExperimentResult, ResearchReport, Contradiction, ResearchGap, ThemeCluster, TokenUsageLog  # noqa: F401
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    _migrate_db(engine)
    log.info("database.initialized", path=str(settings.db_path))


def get_session() -> Session:
    return Session(get_engine())


def init_chroma():
    """Initialize ChromaDB persistent client."""
    import chromadb
    ensure_dirs()
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    log.info("chromadb.initialized", path=str(settings.chroma_path))
    return client


_chroma_client = None


def get_chroma():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = init_chroma()
    return _chroma_client
