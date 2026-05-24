"""One-time data repair for column-rotated paper_analysis rows.

Background: the bug #12 historical-DB merge ran `INSERT ... SELECT *` between two
paper_analysis schemas whose trailing 6 columns were appended in a different order.
The result is a deterministic positional *rotation* of those columns for the merged
rows — the data is intact, just in the wrong slots. The DATETIME `analyzed_at` ends up
holding a difficulty string ('medium'/'easy'/'hard'), which crashes any full-entity load.

This module un-rotates those rows. It is idempotent (rows whose `analyzed_at` is already a
valid timestamp are left untouched) and conservative (a row is only rewritten when the
*recovered* `analyzed_at` is itself a valid timestamp — i.e. it matches the known rotation).

Run once against the live DB:  uv run python -m core.repair
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import text

from core.database import get_engine

log = structlog.get_logger()

# Columns read for rotation detection/repair, in active-schema order.
_ROTATED_COLS = [
    "id", "limitations", "datasets_used", "key_hyperparameters",
    "reproducibility_difficulty", "raw_claude_response", "analyzed_at",
]


def _is_valid_timestamp(value) -> bool:
    if value is None:
        return False
    try:
        datetime.fromisoformat(str(value))
        return True
    except ValueError:
        return False


def repair_paper_analysis(engine=None) -> int:
    """Un-rotate corrupt paper_analysis rows in place. Returns the number of rows repaired.

    Inverse of the observed rotation:
        real raw_claude_response        <- current limitations
        real analyzed_at                <- current datasets_used
        real limitations                <- current key_hyperparameters
        real datasets_used              <- current reproducibility_difficulty
        real key_hyperparameters        <- current raw_claude_response
        real reproducibility_difficulty <- current analyzed_at
    """
    eng = engine or get_engine()
    repaired = 0
    with eng.connect() as conn:
        rows = conn.execute(
            text(f"SELECT {', '.join(_ROTATED_COLS)} FROM paper_analysis")
        ).fetchall()

        for r in rows:
            d = dict(zip(_ROTATED_COLS, r))
            if _is_valid_timestamp(d["analyzed_at"]):
                continue  # already valid — nothing to do (idempotent)

            recovered = {
                "raw_claude_response": d["limitations"],
                "analyzed_at": d["datasets_used"],
                "limitations": d["key_hyperparameters"],
                "datasets_used": d["reproducibility_difficulty"],
                "key_hyperparameters": d["raw_claude_response"],
                "reproducibility_difficulty": d["analyzed_at"],
            }
            # Only rewrite if this matches the known rotation (recovered timestamp is valid).
            if not _is_valid_timestamp(recovered["analyzed_at"]):
                log.warning("repair.unrecoverable_row", row_id=d["id"])
                continue

            conn.execute(
                text(
                    "UPDATE paper_analysis SET "
                    "raw_claude_response = :rcr, analyzed_at = :aa, limitations = :lim, "
                    "datasets_used = :ds, key_hyperparameters = :kh, "
                    "reproducibility_difficulty = :rd "
                    "WHERE id = :id"
                ),
                {
                    "rcr": recovered["raw_claude_response"],
                    "aa": recovered["analyzed_at"],
                    "lim": recovered["limitations"],
                    "ds": recovered["datasets_used"],
                    "kh": recovered["key_hyperparameters"],
                    "rd": recovered["reproducibility_difficulty"],
                    "id": d["id"],
                },
            )
            repaired += 1
        conn.commit()

    log.info("repair.paper_analysis_done", repaired=repaired)
    return repaired


if __name__ == "__main__":
    from core.database import init_db
    init_db()
    count = repair_paper_analysis()
    print(f"Repaired {count} paper_analysis row(s).")
