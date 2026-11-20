"""Find the last date to apply in a job posting.

Sources rarely provide it as a field, so this also reads phrases such as "Application deadline:
December 16, 2026" or "Last date to apply: 30/11/2026" from the description. Only dates that
follow such a phrase are used, so a start date or a founding year is never mistaken for one.
"""

import calendar
import re
from datetime import date, datetime

_PHRASE = re.compile(
    r"(last date (to|for|of) appl\w*|apply (by|before)|application(s)? (deadline|close[sd]?|due)|"
    r"deadline (for|to) appl\w*|closing date|applications? will close|apply no later than)",
    re.IGNORECASE,
)
_MONTHS = (
    {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    | {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
    | {"sept": 9}
)
_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
_PATTERNS = [
    (re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", re.I), "mdy"),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON},?\s+(\d{{4}})", re.I), "dmy"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    (re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b"), "numeric"),  # Indian style: day first
    (re.compile(rf"\b{_MON}\s+(\d{{4}})\b", re.I), "my"),  # "January 2027" -> end of month
]


def _parse(text: str) -> date | None:
    best: tuple[int, date] | None = None
    for pattern, kind in _PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        try:
            if kind == "mdy":
                d = date(int(m.group(3)), _MONTHS[m.group(1).lower().rstrip(".")], int(m.group(2)))
            elif kind == "dmy":
                d = date(int(m.group(3)), _MONTHS[m.group(2).lower().rstrip(".")], int(m.group(1)))
            elif kind == "iso":
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif kind == "numeric":
                d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            else:
                y, mo = int(m.group(2)), _MONTHS[m.group(1).lower().rstrip(".")]
                d = date(y, mo, calendar.monthrange(y, mo)[1])
        except (ValueError, KeyError):
            continue
        if best is None or m.start() < best[0]:  # the date written closest to the phrase wins
            best = (m.start(), d)
    return best[1] if best else None


def find_apply_by(
    description: str, posted: datetime | None = None, field_value: str | None = None
) -> date | None:
    if field_value:
        try:
            return datetime.fromisoformat(field_value.replace("Z", "+00:00")).date()
        except ValueError:
            pass
    for m in _PHRASE.finditer(description or ""):
        d = _parse(description[m.end() : m.end() + 90])
        if d is None:
            continue
        # Plausibility: not before the posting date, and not absurdly far ahead.
        if posted and d < posted.date():
            continue
        if d.year > date.today().year + 2:
            continue
        return d
    return None
