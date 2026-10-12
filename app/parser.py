"""Rule-based information extraction from resume text."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .preprocess import is_heading

# --------------------------------------------------------------------------- skill matching
_NOT_WORD = r"a-z0-9+#"  # characters that make a match part of a longer token (so 'sql' != 'mysql', 'c' != 'c++')


@dataclass(frozen=True)
class Match:
    count: int
    term: str  # the wording that matched first, e.g. "sklearn" for the skill "scikit-learn"


class SkillMatcher:
    """Matches canonical skills and their aliases in lower-cased text using word boundaries.

    'scikit-learn' also matches 'scikit learn' and 'sklearn' (via alias); 'sql' does not match
    inside 'mysql' unless 'mysql' is listed as an alias of SQL; 'c++' and 'node.js' work.
    """

    def __init__(self, aliases: dict[str, tuple[str, ...]]):
        self._aliases = aliases
        self._patterns: dict[str, re.Pattern[str]] = {}

    @staticmethod
    def _term_pattern(term: str) -> str:
        tokens = [t for t in re.split(r"[\s\-]+", term.lower().strip()) if t]
        return r"[\s\-]*".join(re.escape(t) for t in tokens)

    def register(self, label: str) -> None:
        key = label.lower()
        if key in self._patterns:
            return
        terms = [label, *self._aliases.get(key, ())]
        alternatives = "|".join(self._term_pattern(t) for t in sorted(set(terms), key=len, reverse=True))
        self._patterns[key] = re.compile(rf"(?<![{_NOT_WORD}])(?:{alternatives})(?![{_NOT_WORD}])")

    def search(self, label: str, text_lower: str) -> Match | None:
        pattern = self._patterns.get(label.lower())
        if pattern is None:
            self.register(label)
            pattern = self._patterns[label.lower()]
        hits = [m.group(0) for m in pattern.finditer(text_lower)]
        if not hits:
            return None
        return Match(count=len(hits), term=re.sub(r"\s+", " ", hits[0]))


# --------------------------------------------------------------------------- contact details
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RES = (
    re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)"),                  # Indian mobile
    re.compile(r"(?<!\d)\+\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}(?!\d)"),  # international
)
_LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in/|github\.com/)[A-Za-z0-9_\-./%]+", re.I)
_NOT_NAMES = {"resume", "curriculum vitae", "cv", "curriculum vitae cv", "biodata"}


@dataclass
class Contact:
    name: str = ""
    email: str = ""
    phone: str = ""
    links: list[str] = field(default_factory=list)


def parse_contact(text: str) -> Contact:
    contact = Contact()
    if m := _EMAIL_RE.search(text):
        contact.email = m.group(0)
    for pattern in _PHONE_RES:
        if m := pattern.search(text):
            contact.phone = re.sub(r"\s+", " ", m.group(0)).strip()
            break
    seen: set[str] = set()
    for m in _LINK_RE.finditer(text):
        link = m.group(0).rstrip("./")
        if link.lower() not in seen:
            seen.add(link.lower())
            contact.links.append(link)

    for line in [ln.strip() for ln in text.splitlines() if ln.strip()][:6]:
        words = line.split()
        if (
            2 <= len(words) <= 4
            and line.lower() not in _NOT_NAMES
            and not is_heading(line)
            and all(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", w) for w in words)
        ):
            contact.name = line.title() if line.isupper() else line
            break
    return contact


# --------------------------------------------------------------------------- experience
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_DATE = rf"(?:\b{_MONTH}\.?,?\s*(?:19|20)\d{{2}}|\b\d{{1,2}}/(?:19|20)\d{{2}}|\b(?:19|20)\d{{2}}\b)"
_PRESENT = r"(?:present|current|currently|now|ongoing|till\s+date|to\s+date|date)"
_RANGE_RE = re.compile(rf"({_DATE})\s*(?:-|to|until|till)\s*({_DATE}|{_PRESENT}\b)", re.I)
_MONTH_INDEX = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_STATED_RE = re.compile(
    r"(\d{1,2}(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+|relevant\s+|industry\s+|work\s+)?experience",
    re.I,
)
_EDU_LINE_RE = re.compile(
    r"\b(b\.?\s?tech|m\.?\s?tech|b\.\s?e\b|b\.?\s?sc|m\.?\s?sc|bca|mca|mba|diploma|bachelor|master|ph\.?d|"
    r"university|college|institute|school|cgpa|gpa|percentage|polytechnic|sslc|intermediate)\b",
    re.I,
)


@dataclass
class ExperienceInfo:
    years: float = 0.0
    source: str = "none"          # "date ranges" | "stated in resume" | "none"
    stated_years: float = 0.0
    ranges_found: int = 0


def _to_month_index(token: str, *, is_end: bool, today: date) -> tuple[int, bool] | None:
    """Return (month_index, has_explicit_month). Year-only dates default to mid-year."""
    low = token.lower().strip()
    if re.fullmatch(_PRESENT, low):
        return today.year * 12 + today.month - 1, True
    year_match = re.search(r"(?:19|20)\d{2}", low)
    if not year_match:
        return None
    year = int(year_match.group(0))
    if numeric := re.match(r"(\d{1,2})/", low):
        month = int(numeric.group(1))
        return (year * 12 + month - 1, True) if 1 <= month <= 12 else None
    if named := re.match(_MONTH, low):
        return year * 12 + _MONTH_INDEX[named.group(0)[:3]] - 1, True
    return year * 12 + 5, False  # 'June' as a neutral default for year-only dates


def estimate_experience(text: str, sections: dict[str, str], today: date | None = None) -> ExperienceInfo:
    """Total months covered by date ranges in the experience section (overlaps merged), in years."""
    today = today or date.today()
    info = ExperienceInfo()
    if m := _STATED_RE.search(text):
        info.stated_years = float(m.group(1))

    if sections.get("experience"):
        scope = sections["experience"]
    else:  # no experience heading: scan everything except lines that look like education
        scope = "\n".join(ln for ln in text.splitlines() if not _EDU_LINE_RE.search(ln))

    spans: list[tuple[int, int]] = []
    now_index = today.year * 12 + today.month - 1
    for m in _RANGE_RE.finditer(scope):
        start = _to_month_index(m.group(1), is_end=False, today=today)
        end = _to_month_index(m.group(2), is_end=True, today=today)
        if not start or not end:
            continue
        end_exclusive = end[0] + 1 if end[1] else end[0]
        end_exclusive = min(end_exclusive, now_index + 1)
        if end_exclusive > start[0]:
            spans.append((start[0], end_exclusive))
    info.ranges_found = len(spans)

    months, last_end, last_start = 0, None, None
    for start, end in sorted(spans):
        if last_end is None or start > last_end:
            if last_end is not None:
                months += last_end - last_start
            last_start, last_end = start, end
        else:
            last_end = max(last_end, end)
    if last_end is not None:
        months += last_end - last_start

    if months:
        info.years, info.source = round(months / 12, 1), "date ranges"
    elif info.stated_years:
        info.years, info.source = info.stated_years, "stated in resume"
    return info


# --------------------------------------------------------------------------- education
EDUCATION_LABELS = {0: "Not detected", 1: "Diploma", 2: "Bachelor's degree", 3: "Master's degree", 4: "PhD"}
_DEGREE_PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (4, re.compile(r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctor of philosophy\b", re.I)),
    (3, re.compile(r"\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bmca\b|\bmba\b|\bm\.\s?e\b|\bm\.?\s?s\b(?=\s+(?:in|degree))|"
                   r"\bmaster(?:['’]?s)?\s+(?:of|in|degree)\b|\bpost[\s\-]?graduat\w*", re.I)),
    (2, re.compile(r"\bb\.?\s?tech\b|\bb\.?\s?sc\b|\bbca\b|\bb\.?\s?com\b|\bb\.\s?e\b|\bbachelor(?:['’]?s)?\b|"
                   r"\bundergraduate\b|\bb\.?\s?s\b(?=\s+(?:in|degree))", re.I)),
    (1, re.compile(r"\bdiploma\b|\bpolytechnic\b", re.I)),
)
_BE_UPPER_RE = re.compile(r"\bBE\b")  # 'BE' is only trusted when written in capitals


def detect_education(text: str, sections: dict[str, str]) -> int:
    """Highest degree level found (0 if none). Prefers the education section when present."""
    scope = sections.get("education") or text
    best = 0
    for level, pattern in _DEGREE_PATTERNS:
        if pattern.search(scope):
            best = max(best, level)
    if best < 2 and _BE_UPPER_RE.search(scope):
        best = 2
    if best == 0 and scope is not text:  # heading found but degree named elsewhere
        return detect_education(text, {})
    return best
