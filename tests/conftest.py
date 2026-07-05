import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """Redirect settings.data_dir to a per-test tmp dir so tests never write to
    the real data/ directory (state JSON, reports, etc.)."""
    from config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture()
def in_memory_engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    # Import all models to populate SQLModel.metadata
    import core.models  # noqa: F401
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr("core.database._engine", engine)
    yield engine
