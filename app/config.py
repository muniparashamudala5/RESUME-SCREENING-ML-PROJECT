"""Runtime settings. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    roles_path: Path
    static_dir: Path
    max_upload_bytes: int
    max_pages: int
    max_batch_files: int
    min_text_chars: int      # below this many characters we assume a scanned PDF and try OCR
    ocr_enabled: bool
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        roles_path=Path(os.getenv("ROLES_PATH", BASE_DIR / "data" / "roles.json")),
        static_dir=BASE_DIR / "app" / "static",
        max_upload_bytes=_env_int("MAX_UPLOAD_MB", 5) * 1024 * 1024,
        max_pages=_env_int("MAX_PAGES", 10),
        max_batch_files=_env_int("MAX_BATCH_FILES", 20),
        min_text_chars=_env_int("MIN_TEXT_CHARS", 200),
        ocr_enabled=_env_bool("OCR_ENABLED", True),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
