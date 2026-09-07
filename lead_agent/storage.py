"""
Persistent storage for lead state, so a lead's conversation survives across
messages (and across app restarts) instead of living only in one Python
process's memory. One row per lead_id, state stored as JSON -- simple and
sufficient for this scale; a real deployment would normalize this out.
"""

import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "lead_agent.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS lead_state (
    lead_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at_day INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sim_clock (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    day INTEGER NOT NULL
);
"""


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO sim_clock (id, day) VALUES (1, 0)"
    )
    conn.commit()
    return conn


def load_state(lead_id):
    conn = _conn()
    row = conn.execute(
        "SELECT state_json FROM lead_state WHERE lead_id=?",
        (lead_id,),
    ).fetchone()
    conn.close()

    return json.loads(row[0]) if row else None


def save_state(lead_id, state: dict, day: int):
    conn = _conn()

    conn.execute(
        """
        INSERT INTO lead_state
        (lead_id, state_json, updated_at_day)
        VALUES (?,?,?)
        ON CONFLICT(lead_id)
        DO UPDATE SET
            state_json=excluded.state_json,
            updated_at_day=excluded.updated_at_day
        """,
        (
            lead_id,
            json.dumps(state),
            day,
        ),
    )

    conn.commit()
    conn.close()


def all_lead_ids_with_state():
    conn = _conn()

    rows = conn.execute(
        "SELECT lead_id FROM lead_state"
    ).fetchall()

    conn.close()

    return [r[0] for r in rows]


def get_sim_day():
    conn = _conn()

    row = conn.execute(
        "SELECT day FROM sim_clock WHERE id=1"
    ).fetchone()

    conn.close()

    return row[0]


def set_sim_day(day: int):
    conn = _conn()

    conn.execute(
        "UPDATE sim_clock SET day=? WHERE id=1",
        (day,),
    )

    conn.commit()
    conn.close()


def reset_all():
    conn = _conn()

    conn.execute("DELETE FROM lead_state")

    conn.execute(
        "UPDATE sim_clock SET day=0 WHERE id=1"
    )

    conn.commit()
    conn.close()