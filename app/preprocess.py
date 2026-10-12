"""Text cleaning and resume section detection."""
from __future__ import annotations

import re
import unicodedata

BULLETS = "•●○◦▪▫■□◆◇➢➤►▶✓✔★☆–—‣⁃∙·"
_BULLET_RE = re.compile(f"[{re.escape(BULLETS)}]")
_DASHES_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2212]")

# heading text (normalised) -> canonical section name
_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "career summary", "profile", "professional profile",
                "objective", "career objective", "about me", "about"),
    "skills": ("skills", "technical skills", "key skills", "core skills", "core competencies", "skill set",
               "technical proficiency", "technologies", "tech stack", "tools and technologies",
               "areas of expertise", "technical expertise"),
    "experience": ("experience", "work experience", "professional experience", "employment history",
                   "work history", "career history", "internship", "internships", "internship experience",
                   "industrial experience", "experience and internships", "relevant experience"),
    "projects": ("projects", "academic projects", "personal projects", "key projects", "project work",
                 "selected projects", "technical projects"),
    "education": ("education", "academic background", "academic qualifications", "educational qualifications",
                  "qualifications", "academics", "educational background"),
    "certifications": ("certifications", "certificates", "courses", "licenses and certifications",
                       "certifications and courses", "training", "courses and certifications"),
    "other": ("achievements", "awards", "honors", "honours", "accomplishments", "extracurricular activities",
              "activities", "publications", "languages", "interests", "hobbies", "references", "declaration",
              "personal details", "personal information"),
}
_HEADING_TO_SECTION = {alias: name for name, aliases in _SECTION_ALIASES.items() for alias in aliases}

# sections whose text counts as evidence that a skill was actually used
EVIDENCE_SECTIONS = ("experience", "projects", "certifications")


def clean_text(raw: str) -> str:
    """Normalise unicode, bullets, dashes and whitespace while keeping line breaks."""
    text = unicodedata.normalize("NFKC", raw)
    text = re.sub(r"\(cid:\d+\)", "- ", text)      # unmapped glyphs, usually bullets, from some PDF fonts
    text = text.replace("\u00a0", " ").replace("\r", "\n")
    text = _DASHES_RE.sub("-", text)
    text = _BULLET_RE.sub("- ", text)
    text = re.sub(r"[^\S\n]+", " ", text)          # collapse runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _heading_key(line: str) -> str:
    key = line.lower().replace("&", " and ")
    key = re.sub(r"[^a-z\s]", " ", key)             # drop digits, punctuation, decorative rules
    return re.sub(r"\s+", " ", key).strip()


def is_heading(line: str) -> bool:
    return len(line) <= 45 and _HEADING_TO_SECTION.get(_heading_key(line.strip())) is not None


def split_sections(text: str) -> dict[str, str]:
    """Split a resume into sections keyed by canonical name.

    Text before the first recognised heading is stored under "header". Returns
    {"header": whole_text} when no headings are found, so callers can tell detection failed.
    """
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    found_any = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            sections[current].append("")
            continue

        heading, inline = None, ""
        if len(stripped) <= 45:
            heading = _HEADING_TO_SECTION.get(_heading_key(stripped))
        if heading is None and ":" in stripped:      # "Skills: Python, SQL, Excel"
            head, _, rest = stripped.partition(":")
            candidate = _HEADING_TO_SECTION.get(_heading_key(head))
            if candidate and len(head.split()) <= 4 and rest.strip():
                heading, inline = candidate, rest.strip()

        if heading:
            found_any = True
            current = heading
            sections.setdefault(current, [])
            if inline:
                sections[current].append(inline)
        else:
            sections[current].append(stripped)

    if not found_any:
        return {"header": text}
    return {name: "\n".join(lines).strip() for name, lines in sections.items() if "".join(lines).strip()}
