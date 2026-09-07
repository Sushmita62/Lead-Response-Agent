"""Retry/nurture cadence for leads who go quiet or who explicitly aren't
ready yet. Pure scheduling math, called from the nurture node and from the
simulated-clock sweep in app.py."""

CADENCE_DAYS = [1, 3, 7, 21, 30, 30]
MAX_ATTEMPTS = len(CADENCE_DAYS)


def days_until_next(attempt_count: int) -> int:
    idx = min(attempt_count, MAX_ATTEMPTS - 1)
    return CADENCE_DAYS[idx]


def should_stop(attempt_count: int, explicit_stop: bool) -> bool:
    return explicit_stop or attempt_count >= MAX_ATTEMPTS


def is_due(nurture: dict, sim_day: int) -> bool:
    if not nurture or nurture.get("stopped"):
        return False
    next_contact_at = nurture.get("next_contact_at")
    if next_contact_at is None:
        return False
    return sim_day >= next_contact_at
