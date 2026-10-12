"""Loads and validates role profiles (skills, keywords, scoring rules) from JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

WEIGHT_KEYS = ("must_have", "good_to_have", "keywords", "similarity", "experience", "education")


@dataclass(frozen=True)
class Requirement:
    """One thing a role asks for. `options` has several entries for 'any of' requirements."""

    label: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class Role:
    id: str
    title: str
    description: str
    must_have: tuple[Requirement, ...]
    good_to_have: tuple[Requirement, ...]
    keywords: tuple[Requirement, ...]
    min_experience_years: float
    min_education_level: int


@dataclass(frozen=True)
class ScoringConfig:
    weights: dict[str, float]
    declared_only_credit: float
    similarity_saturation: float
    shortlist_score: float
    review_score: float
    shortlist_min_must_coverage: float
    review_min_must_coverage: float


@dataclass(frozen=True)
class Profiles:
    roles: dict[str, Role]
    aliases: dict[str, tuple[str, ...]]  # lower-cased canonical label -> alternative spellings
    scoring: ScoringConfig


def _requirement(entry: str | list[str], where: str) -> Requirement:
    if isinstance(entry, str) and entry.strip():
        return Requirement(label=entry.strip(), options=(entry.strip(),))
    if isinstance(entry, list) and entry and all(isinstance(o, str) and o.strip() for o in entry):
        options = tuple(o.strip() for o in entry)
        return Requirement(label=" / ".join(options), options=options)
    raise ValueError(f"{where}: each skill must be a non-empty string or a non-empty list of strings")


def _requirements(items: list, where: str) -> tuple[Requirement, ...]:
    if not isinstance(items, list):
        raise ValueError(f"{where} must be a list")
    return tuple(_requirement(item, where) for item in items)


def _scoring(raw: dict) -> ScoringConfig:
    weights = {k: float(raw["weights"][k]) for k in WEIGHT_KEYS}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"scoring.weights must add up to 1.0 (got {total:.3f})")
    t = raw["thresholds"]
    config = ScoringConfig(
        weights=weights,
        declared_only_credit=float(raw["declared_only_credit"]),
        similarity_saturation=float(raw["similarity_saturation"]),
        shortlist_score=float(t["shortlist_score"]),
        review_score=float(t["review_score"]),
        shortlist_min_must_coverage=float(t["shortlist_min_must_coverage"]),
        review_min_must_coverage=float(t["review_min_must_coverage"]),
    )
    if config.review_score > config.shortlist_score:
        raise ValueError("scoring.thresholds: review_score must not exceed shortlist_score")
    if not 0 < config.similarity_saturation <= 1:
        raise ValueError("scoring.similarity_saturation must be in (0, 1]")
    return config


def load_profiles(path: Path | str) -> Profiles:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    roles: dict[str, Role] = {}
    for role_id, spec in raw["roles"].items():
        where = f"roles.{role_id}"
        must_have = _requirements(spec["must_have"], f"{where}.must_have")
        if not must_have:
            raise ValueError(f"{where}.must_have cannot be empty")
        roles[role_id] = Role(
            id=role_id,
            title=spec["title"],
            description=spec.get("description", ""),
            must_have=must_have,
            good_to_have=_requirements(spec.get("good_to_have", []), f"{where}.good_to_have"),
            keywords=_requirements(spec.get("keywords", []), f"{where}.keywords"),
            min_experience_years=float(spec.get("min_experience_years", 0)),
            min_education_level=int(spec.get("min_education_level", 0)),
        )
    if not roles:
        raise ValueError("roles.json defines no roles")

    aliases = {k.lower(): tuple(v) for k, v in raw.get("aliases", {}).items()}
    return Profiles(roles=roles, aliases=aliases, scoring=_scoring(raw["scoring"]))
