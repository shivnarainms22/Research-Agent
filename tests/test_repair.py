"""Tests for core/repair.py — un-rotating column-corrupt paper_analysis rows.

The live DB inherited 23 paper_analysis rows from the bug #12 historical merge whose
trailing 6 columns are positionally rotated (INSERT...SELECT * across mismatched schemas).
The rotation is deterministic and fully reversible — see core/repair.py.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import text
from sqlmodel import Session, select


# Active-schema rotation observed in the live DB. A corrupt row's columns hold:
#   limitations                <- real raw_claude_response
#   datasets_used              <- real analyzed_at (a timestamp)
#   key_hyperparameters        <- real limitations
#   reproducibility_difficulty <- real datasets_used
#   raw_claude_response        <- real key_hyperparameters
#   analyzed_at                <- real reproducibility_difficulty ('medium'/'easy'/'hard')
def _insert_corrupt_row(engine, paper_id="pc", row_id="ac"):
    with engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO paper_analysis "
            "(id, paper_id, key_contributions, methods_described, reproducible_experiments, "
            " novelty_score, relevance_score, limitations, datasets_used, key_hyperparameters, "
            " reproducibility_difficulty, raw_claude_response, analyzed_at) "
            "VALUES (:id, :pid, :kc, :md, :re, 7.5, 9.0, :lim, :ds, :kh, :rd, :rcr, :aa)"
        ), {
            "id": row_id, "pid": paper_id,
            "kc": '["contribution A"]', "md": '["method B"]', "re": '[{"title": "Exp"}]',
            # rotated values:
            "lim": '{"REAL_RAW": 1}',                 # -> raw_claude_response
            "ds": '2026-03-08 23:23:59.535198',       # -> analyzed_at (valid ts)
            "kh": '["REAL_LIM"]',                     # -> limitations
            "rd": '["REAL_DS"]',                       # -> datasets_used
            "rcr": '{"REAL_KH": 2}',                  # -> key_hyperparameters
            "aa": 'hard',                              # -> reproducibility_difficulty
        })
        conn.commit()


def _insert_clean_row(engine, paper_id="pk", row_id="ak"):
    from core.models import PaperAnalysis
    with Session(engine, expire_on_commit=False) as session:
        session.add(PaperAnalysis(
            id=row_id, paper_id=paper_id, key_contributions='["x"]',
            novelty_score=8.0, relevance_score=8.0,
            reproducibility_difficulty="easy", analyzed_at=datetime(2026, 1, 2, 3, 4, 5),
        ))
        session.commit()


def test_repair_unrotates_corrupt_row(in_memory_engine):
    from core.repair import repair_paper_analysis
    from core.models import PaperAnalysis

    _insert_corrupt_row(in_memory_engine)
    n = repair_paper_analysis(in_memory_engine)
    assert n == 1

    # The corrupt row now loads as a full ORM entity without raising, with recovered values.
    with Session(in_memory_engine) as session:
        row = session.exec(select(PaperAnalysis).where(PaperAnalysis.id == "ac")).one()
    assert row.raw_claude_response == '{"REAL_RAW": 1}'
    assert row.analyzed_at == datetime(2026, 3, 8, 23, 23, 59, 535198)
    assert row.limitations == '["REAL_LIM"]'
    assert row.datasets_used == '["REAL_DS"]'
    assert row.key_hyperparameters == '{"REAL_KH": 2}'
    assert row.reproducibility_difficulty == "hard"
    # Intact fields untouched.
    assert row.key_contributions == '["contribution A"]'
    assert row.novelty_score == 7.5


def test_repair_is_idempotent(in_memory_engine):
    from core.repair import repair_paper_analysis
    _insert_corrupt_row(in_memory_engine)
    assert repair_paper_analysis(in_memory_engine) == 1
    assert repair_paper_analysis(in_memory_engine) == 0  # already valid, nothing to do


def test_repair_leaves_clean_rows_untouched(in_memory_engine):
    from core.repair import repair_paper_analysis
    from core.models import PaperAnalysis
    _insert_clean_row(in_memory_engine)
    assert repair_paper_analysis(in_memory_engine) == 0
    with Session(in_memory_engine) as session:
        row = session.exec(select(PaperAnalysis).where(PaperAnalysis.id == "ak")).one()
    assert row.reproducibility_difficulty == "easy"
    assert row.analyzed_at == datetime(2026, 1, 2, 3, 4, 5)


def test_repair_skips_unrecoverable_rows(in_memory_engine):
    """A row corrupt in an unknown way (recovered analyzed_at also not a timestamp) is left alone."""
    from core.repair import repair_paper_analysis
    with in_memory_engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO paper_analysis "
            "(id, paper_id, key_contributions, methods_described, reproducible_experiments, "
            " novelty_score, relevance_score, limitations, datasets_used, key_hyperparameters, "
            " reproducibility_difficulty, raw_claude_response, analyzed_at) "
            "VALUES ('weird','pw','[]','[]','[]',0,0,'[]','not-a-date','{}','[]','{}','medium')"
        ))
        conn.commit()
    assert repair_paper_analysis(in_memory_engine) == 0  # datasets_used isn't a timestamp -> skip


def test_get_analysis_survives_corrupt_row(in_memory_engine):
    """Harden: get_analysis must not crash on an unrepaired corrupt row; intact fields returned."""
    from knowledge.paper_store import get_analysis
    _insert_corrupt_row(in_memory_engine, paper_id="pc2", row_id="ac2")
    analysis = get_analysis("pc2")  # must not raise
    assert analysis is not None
    assert analysis.key_contributions == '["contribution A"]'
    assert analysis.novelty_score == 7.5
