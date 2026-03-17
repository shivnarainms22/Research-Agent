import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel


@pytest.fixture()
def in_memory_engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    # Import all models to populate SQLModel.metadata
    import core.models  # noqa: F401
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr("core.database._engine", engine)
    yield engine
