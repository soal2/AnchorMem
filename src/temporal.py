"""Validate source-grounded ISO time values without inventing precision or zones."""

import re
from calendar import monthrange
from datetime import datetime

TEMPORAL_FIELDS = ("occurred_start", "occurred_end", "occurred_precision", "mentioned_at")
PRECISIONS = ("year", "month", "day", "hour", "minute", "second")
_DATE = re.compile(
    r"^\d{4}(?:-\d{2}(?:-\d{2}(?:T(?P<hour>\d{2})(?::(?P<minute>\d{2})"
    r"(?::(?P<second>\d{2})(?:\.(?P<fraction>\d+))?)?)?(?:Z|[+-]\d{2}:\d{2})?)?)?)?$"
)
ISO_TIME_PATTERN = re.sub(r"\(\?P<[^>]+>", "(?:", _DATE.pattern)


def _parse(value: str) -> datetime:
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        raise ValueError("time must be an ISO year, month, date, or datetime")
    if len(value) == 4:
        value += "-01-01"
    elif len(value) == 7:
        value += "-01"
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def time_bounds(value: str) -> tuple[datetime, datetime]:
    """Return inclusive comparison bounds at the precision explicitly supplied.

    These bounds are only for comparison; callers retain the original source text.
    No timezone is added, and naive/aware bounds must not be compared together.
    """
    lower = _parse(value)
    match = _DATE.fullmatch(value)
    if len(value) == 4:
        upper = lower.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
    elif len(value) == 7:
        upper = lower.replace(day=monthrange(lower.year, lower.month)[1], hour=23, minute=59, second=59, microsecond=999999)
    elif match.group("hour") is None:
        upper = lower.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif match.group("minute") is None:
        upper = lower.replace(minute=59, second=59, microsecond=999999)
    elif match.group("second") is None:
        upper = lower.replace(second=59, microsecond=999999)
    elif match.group("fraction") is None:
        upper = lower.replace(microsecond=999999)
    else:
        quantum = 10 ** max(0, 6 - len(match.group("fraction")))
        upper = lower.replace(microsecond=lower.microsecond + quantum - 1)
    return lower, upper


def validate_temporal(fact: dict) -> dict:
    # Some strict JSON providers emit the literal string "null" for an unknown
    # date. It has the same unambiguous meaning as JSON null, never a date.
    result = {key: None if isinstance(fact.get(key), str) and fact[key].strip().lower() == "null"
              else fact.get(key) for key in TEMPORAL_FIELDS}
    precision = result["occurred_precision"]
    if precision is not None and precision not in PRECISIONS:
        raise ValueError("invalid occurred_precision")
    parsed = {key: time_bounds(result[key]) for key in ("occurred_start", "occurred_end", "mentioned_at")
              if result[key] is not None}
    if (result["occurred_start"] is not None or result["occurred_end"] is not None) != (precision is not None):
        raise ValueError("event time and occurred_precision must be supplied together")
    start, end = parsed.get("occurred_start"), parsed.get("occurred_end")
    if start is not None and end is not None:
        # A naive timestamp does not identify an instant comparable to an aware one.
        start_lower, end_upper = start[0], end[1]
        if (start_lower.utcoffset() is None) == (end_upper.utcoffset() is None) and end_upper < start_lower:
            raise ValueError("occurred_end precedes occurred_start")
    return result


def event_overlaps(fact, time_range) -> bool:
    """Match source event intervals against a half-open query without adding zones."""
    if time_range is None:
        return True
    start, end = fact["occurred_start"], fact["occurred_end"]
    if start is None and end is None:
        return False
    lower = time_bounds(start)[0] if start is not None else None
    upper = time_bounds(end)[1] if end is not None else None
    query_start = time_bounds(time_range.start)[0] if time_range.start else None
    query_end = time_bounds(time_range.end)[0] if time_range.end else None
    known = [value for value in (lower, upper, query_start, query_end) if value is not None]
    if len({value.utcoffset() is not None for value in known}) > 1:
        return False
    return ((query_end is None or lower is None or lower < query_end)
            and (query_start is None or upper is None or upper >= query_start))
