"""Screening engine: resume text + role profile -> explainable score and decision."""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings
from .extractor import ExtractionError, ExtractionResult, extract_text
from .parser import EDUCATION_LABELS, SkillMatcher, detect_education, estimate_experience, parse_contact
from .preprocess import EVIDENCE_SECTIONS, clean_text, split_sections
from .roles import Profiles, Requirement, Role, load_profiles

logger = logging.getLogger(__name__)

STUFFING_THRESHOLD = 12  # one skill mentioned this many times is suspicious
SHORT_TEXT_CHARS = 300

COMPONENT_LABELS = {
    "must_have": "Must-have skills",
    "good_to_have": "Good-to-have skills",
    "keywords": "Role keywords",
    "similarity": "Overall text similarity",
    "experience": "Experience",
    "education": "Education",
}


class UnknownRoleError(KeyError):
    """The requested role id does not exist in roles.json."""


class Screener:
    def __init__(self, profiles: Profiles, settings: Settings):
        self.profiles = profiles
        self.settings = settings
        self.matcher = SkillMatcher(profiles.aliases)

        self._vocabulary: list[str] = []
        for role in profiles.roles.values():
            for req in (*role.must_have, *role.good_to_have, *role.keywords):
                for option in req.options:
                    if option not in self._vocabulary:
                        self._vocabulary.append(option)
                    self.matcher.register(option)

        # TF-IDF is fitted once on the role documents; each resume is compared against its role.
        self._role_docs = {rid: self._role_document(role) for rid, role in profiles.roles.items()}
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._vectorizer.fit(list(self._role_docs.values()))
        self._role_vectors = {
            rid: self._vectorizer.transform([doc]) for rid, doc in self._role_docs.items()
        }

    @classmethod
    def from_settings(cls, settings: Settings) -> "Screener":
        return cls(load_profiles(settings.roles_path), settings)

    # ------------------------------------------------------------------ public API
    @property
    def roles(self) -> dict[str, Role]:
        return self.profiles.roles

    def list_roles(self) -> list[dict[str, Any]]:
        return [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "must_have": [q.label for q in r.must_have],
                "good_to_have": [q.label for q in r.good_to_have],
                "keywords": [q.label for q in r.keywords],
                "min_experience_years": r.min_experience_years,
                "min_education": EDUCATION_LABELS.get(r.min_education_level, "Not detected"),
            }
            for r in self.roles.values()
        ]

    def screen_pdf(self, data: bytes, role_id: str, job_description: str = "") -> dict[str, Any]:
        self._role(role_id)  # fail fast on an unknown role, before the expensive extraction
        extraction = extract_text(
            data,
            max_pages=self.settings.max_pages,
            min_chars=self.settings.min_text_chars,
            ocr_enabled=self.settings.ocr_enabled,
        )
        return self.screen_text(extraction.text, role_id, job_description, extraction=extraction)

    def screen_many(self, files: list[tuple[str, bytes]], role_id: str) -> list[dict[str, Any]]:
        """Screen several PDFs for one role and return them ranked, best first. Errors go last."""
        self._role(role_id)
        ranked, failed = [], []
        for filename, data in files:
            try:
                r = self.screen_pdf(data, role_id)
            except ExtractionError as exc:
                failed.append({"filename": filename, "error": str(exc)})
                continue
            ranked.append({
                "filename": filename,
                "name": r["candidate"]["name"],
                "email": r["candidate"]["email"],
                "score": r["score"],
                "decision": r["decision"],
                "must_have_matched": r["summary"]["must_have_matched"],
                "must_have_total": r["summary"]["must_have_total"],
                "missing_must_have": r["skills"]["must_have"]["missing"],
            })
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked + failed

    def screen_text(
        self,
        raw_text: str,
        role_id: str,
        job_description: str = "",
        *,
        extraction: ExtractionResult | None = None,
        today: date | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        role = self._role(role_id)
        cfg = self.profiles.scoring

        text = clean_text(raw_text)
        lower = text.lower()
        sections = split_sections(text)
        section_lower = {name: body.lower() for name, body in sections.items()}
        has_sections = set(sections) != {"header"}
        warnings: list[str] = list(extraction.warnings) if extraction else []

        contact = parse_contact(text)
        experience = estimate_experience(text, sections, today)
        education_level = detect_education(text, sections)

        # --- skills ---------------------------------------------------------------
        must = self._evaluate(role.must_have, lower, section_lower, has_sections)
        good = self._evaluate(role.good_to_have, lower, section_lower, has_sections)
        keys = self._evaluate(role.keywords, lower, section_lower, has_sections)

        # --- similarity, experience, education -------------------------------------
        similarity = self._similarity(lower, role_id, job_description)
        similarity_ratio = min(similarity / cfg.similarity_saturation, 1.0)

        if role.min_experience_years <= 0:
            experience_ratio = 1.0
        else:
            experience_ratio = min(experience.years / role.min_experience_years, 1.0)

        if role.min_education_level <= 0 or education_level >= role.min_education_level:
            education_ratio = 1.0
        elif education_level > 0 and education_level == role.min_education_level - 1:
            education_ratio = 0.5
        else:
            education_ratio = 0.0

        ratios = {
            "must_have": must["credit"],
            "good_to_have": good["credit"],
            "keywords": keys["credit"],
            "similarity": similarity_ratio,
            "experience": experience_ratio,
            "education": education_ratio,
        }
        details = {
            "must_have": f"{must['matched']} of {must['total']} requirements found",
            "good_to_have": f"{good['matched']} of {good['total']} found",
            "keywords": f"{keys['matched']} of {keys['total']} found",
            "similarity": f"text similarity {similarity:.2f} (full marks at {cfg.similarity_saturation:.2f})",
            "experience": (
                f"{experience.years:g} years found, role asks for {role.min_experience_years:g}"
                if role.min_experience_years > 0 else f"{experience.years:g} years found, none required"
            ),
            "education": f"{EDUCATION_LABELS[education_level]} found",
        }
        breakdown = []
        for key, weight in cfg.weights.items():
            breakdown.append({
                "key": key,
                "label": COMPONENT_LABELS[key],
                "max": round(weight * 100, 1),
                "earned": round(weight * 100 * ratios[key], 1),
                "detail": details[key],
            })
        score = round(sum(item["earned"] for item in breakdown), 1)

        # --- decision ---------------------------------------------------------------
        must_coverage = must["matched"] / must["total"]
        if score >= cfg.shortlist_score and must_coverage >= cfg.shortlist_min_must_coverage:
            decision = "Shortlisted"
        elif score >= cfg.review_score and must_coverage >= cfg.review_min_must_coverage:
            decision = "Needs review"
        else:
            decision = "Not shortlisted"

        # --- explanations and warnings -------------------------------------------------
        reasons = self._reasons(decision, score, must, good, experience.years, role, education_level, cfg)
        if not has_sections:
            warnings.append(
                "No standard headings (Skills, Experience, Education) were found, so skills could not be "
                "checked against where they appear."
            )
            if not contact.email:
                warnings.append("This document does not look like a typical resume.")
        if len(text) < SHORT_TEXT_CHARS:
            warnings.append("Very little text was found in this file, so the score may be unreliable.")
        if education_level == 0:
            warnings.append("No degree or diploma was detected.")
        stuffed = [h["label"] for group in (must, good, keys) for h in group["hits"] if h["count"] >= STUFFING_THRESHOLD]
        if stuffed:
            warnings.append(f"Possible keyword stuffing: {', '.join(stuffed)} repeated many times.")

        role_labels = {o.lower() for req in (*role.must_have, *role.good_to_have, *role.keywords) for o in req.options}
        detected = sorted(o for o in self._vocabulary if self.matcher.search(o, lower))

        return {
            "role": {"id": role.id, "title": role.title},
            "score": score,
            "decision": decision,
            "summary": {
                "must_have_matched": must["matched"],
                "must_have_total": must["total"],
                "must_have_coverage": round(must_coverage, 2),
            },
            "breakdown": breakdown,
            "skills": {
                "must_have": {"matched": must["hits"], "missing": must["missing"]},
                "good_to_have": {"matched": good["hits"], "missing": good["missing"]},
                "keywords": {"matched": keys["hits"], "missing": keys["missing"]},
                "other_detected": [s for s in detected if s.lower() not in role_labels],
            },
            "candidate": {
                "name": contact.name,
                "email": contact.email,
                "phone": contact.phone,
                "links": contact.links,
                "experience_years": experience.years,
                "experience_source": experience.source,
                "education": EDUCATION_LABELS[education_level],
            },
            "reasons": reasons,
            "warnings": warnings,
            "meta": {
                "pages": extraction.pages if extraction else None,
                "extraction": extraction.method if extraction else "text",
                "characters": len(text),
                "sections_found": [s for s in sections if s != "header"],
                "processing_ms": round((time.perf_counter() - started) * 1000),
            },
        }

    # ------------------------------------------------------------------ internals
    def _role(self, role_id: str) -> Role:
        try:
            return self.profiles.roles[role_id]
        except KeyError:
            raise UnknownRoleError(role_id) from None

    @staticmethod
    def _role_document(role: Role) -> str:
        parts = [role.title, role.description]
        for req in (*role.must_have, *role.good_to_have, *role.keywords):
            parts.extend(req.options)
        return ". ".join(parts)

    def _similarity(self, lower_text: str, role_id: str, job_description: str) -> float:
        resume_vec = self._vectorizer.transform([lower_text])
        if job_description.strip():
            role_vec = self._vectorizer.transform([f"{self._role_docs[role_id]}. {job_description}"])
        else:
            role_vec = self._role_vectors[role_id]
        return float((resume_vec @ role_vec.T).toarray()[0, 0])

    def _evidence(self, option: str, section_lower: dict[str, str], has_sections: bool) -> str:
        """demonstrated: used in experience/projects/certifications; listed: only declared in the
        skills or summary section; mentioned: found elsewhere (or headings could not be detected)."""
        if not has_sections:
            return "mentioned"
        where = {name for name, body in section_lower.items() if self.matcher.search(option, body)}
        if where & set(EVIDENCE_SECTIONS):
            return "demonstrated"
        if where and where <= {"skills", "summary"}:
            return "listed"
        return "mentioned"

    def _evaluate(
        self,
        requirements: tuple[Requirement, ...],
        lower: str,
        section_lower: dict[str, str],
        has_sections: bool,
    ) -> dict[str, Any]:
        declared_credit = self.profiles.scoring.declared_only_credit
        hits: list[dict[str, Any]] = []
        missing: list[str] = []
        credit_sum = 0.0

        for req in requirements:
            found = []
            for option in req.options:
                m = self.matcher.search(option, lower)
                if m:
                    found.append((option, m, self._evidence(option, section_lower, has_sections)))
            if not found:
                missing.append(req.label)
                continue
            best = max(found, key=lambda f: (f[2] != "listed", f[1].count))
            credit = declared_credit if best[2] == "listed" else 1.0
            credit_sum += credit
            hits.append({
                "label": req.label,
                "matched_term": ", ".join(f[1].term for f in found),
                "evidence": best[2],
                "count": sum(f[1].count for f in found),
                "credit": credit,
            })

        total = len(requirements)
        return {
            "hits": hits,
            "missing": missing,
            "matched": len(hits),
            "total": total,
            "credit": credit_sum / total if total else 1.0,
        }

    @staticmethod
    def _reasons(decision, score, must, good, years, role, education_level, cfg) -> list[str]:
        reasons = []
        reasons.append(f"Matches {must['matched']} of {must['total']} must-have requirements for {role.title}.")
        if must["missing"]:
            reasons.append("Missing must-have: " + ", ".join(must["missing"]) + ".")
        listed_only = [h["label"] for h in must["hits"] + good["hits"] if h["evidence"] == "listed"]
        if listed_only:
            reasons.append(
                "Listed in skills but not shown in experience or projects (credited at "
                f"{int(cfg.declared_only_credit * 100)}%): " + ", ".join(listed_only) + "."
            )
        if role.min_experience_years > 0 and years < role.min_experience_years:
            reasons.append(f"Experience ({years:g} years) is below the {role.min_experience_years:g} year(s) this role asks for.")
        if role.min_education_level and education_level < role.min_education_level:
            reasons.append("Education level is below what this role asks for.")

        if decision == "Shortlisted":
            reasons.append(f"Score {score} meets the {cfg.shortlist_score:g} shortlist threshold.")
        elif decision == "Needs review":
            reasons.append(
                f"Score {score} is between the review ({cfg.review_score:g}) and shortlist "
                f"({cfg.shortlist_score:g}) thresholds, or must-have coverage is below "
                f"{int(cfg.shortlist_min_must_coverage * 100)}%. A person should take a look."
            )
        else:
            reasons.append(f"Score {score} is below the review threshold, or too many must-have skills are missing.")
        return reasons
