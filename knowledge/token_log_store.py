"""CRUD for TokenUsageLog table — persistent per-cycle token accounting."""
from __future__ import annotations

import uuid

from sqlmodel import Session, select

from core.database import get_engine
from core.models import TokenUsageLog

COST_PER_INPUT_TOKEN = 3 / 1_000_000
COST_PER_OUTPUT_TOKEN = 15 / 1_000_000


def save_log(cycle_id: str, module: str, input_tokens: int, output_tokens: int) -> None:
    cost = input_tokens * COST_PER_INPUT_TOKEN + output_tokens * COST_PER_OUTPUT_TOKEN
    log_entry = TokenUsageLog(
        id=str(uuid.uuid4()),
        cycle_id=cycle_id,
        module=module,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )
    with Session(get_engine(), expire_on_commit=False) as session:
        session.add(log_entry)
        session.commit()


def get_logs_for_cycle(cycle_id: str) -> list[TokenUsageLog]:
    with Session(get_engine()) as session:
        return list(
            session.exec(
                select(TokenUsageLog).where(TokenUsageLog.cycle_id == cycle_id)
            ).all()
        )


def get_all_logs(limit: int = 500) -> list[TokenUsageLog]:
    with Session(get_engine()) as session:
        return list(
            session.exec(
                select(TokenUsageLog)
                .order_by(TokenUsageLog.recorded_at.desc())
                .limit(limit)
            ).all()
        )


def get_module_totals() -> dict:
    with Session(get_engine()) as session:
        rows = list(session.exec(select(TokenUsageLog)).all())
    totals: dict = {}
    for row in rows:
        if row.module not in totals:
            totals[row.module] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        totals[row.module]["input_tokens"] += row.input_tokens
        totals[row.module]["output_tokens"] += row.output_tokens
        totals[row.module]["cost_usd"] += row.cost_usd
    return totals
