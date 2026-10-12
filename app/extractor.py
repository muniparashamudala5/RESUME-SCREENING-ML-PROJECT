"""PDF -> text.

Strategy (first one that yields enough text wins):
1. pdfplumber, with a small column-aware reader so two-column resume templates are read
   column by column instead of line by line across the page.
2. pypdf, as a fallback for PDFs pdfplumber struggles with.
3. OCR (pdfplumber page render + Tesseract) for scanned / image-only resumes.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)

MIN_GUTTER_PT = 10.0       # narrowest empty strip we accept as a column gutter
MIN_COLUMN_SHARE = 0.15    # each side of a gutter must hold at least this share of the words
LINE_TOLERANCE_PT = 3.0    # words whose tops differ by less than this sit on the same line


class ExtractionError(Exception):
    """The PDF could not be turned into usable text. The message is safe to show to users."""


@dataclass
class ExtractionResult:
    text: str
    pages: int
    method: str  # "pdfplumber" | "pypdf" | "ocr"
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- validation
def _check_pdf(data: bytes) -> None:
    if not data:
        raise ExtractionError("The uploaded file is empty.")
    if b"%PDF" not in data[:1024]:
        raise ExtractionError("This file is not a PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception:  # noqa: BLE001 - any failure means we cannot open it
                unlocked = 0
            if not unlocked:
                raise ExtractionError(
                    "This PDF is password-protected. Remove the password and upload it again."
                )
        if len(reader.pages) == 0:
            raise ExtractionError("This PDF has no pages.")
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError("This PDF is damaged or could not be opened.") from exc


# --------------------------------------------------------------------------- pdfplumber
def _cluster_lines(words: list[dict]) -> list[list[dict]]:
    """Group words into visual lines (top to bottom), each line sorted left to right."""
    lines: list[list[dict]] = []
    line_tops: list[float] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(word["top"] - line_tops[-1]) <= LINE_TOLERANCE_PT:
            lines[-1].append(word)
        else:
            lines.append([word])
            line_tops.append(word["top"])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def _words_to_text(words: list[dict]) -> str:
    return "\n".join(" ".join(w["text"] for w in line) for line in _cluster_lines(words))


def _find_gutter(words: list[dict], x_min: float, x_max: float) -> tuple[float, float] | None:
    """Return (centre_x, width) of a vertical gutter separating two text columns, or None.

    A gutter is a wide strip that almost no word crosses (a few header words may), with a
    substantial share of the page's words on each side. Single-column pages never qualify
    because words on every line cross any x position between the margins.
    """
    width = x_max - x_min
    total = len(words)
    allowed_crossings = max(1, int(0.03 * total))

    runs: list[tuple[float, float]] = []
    start: float | None = None
    previous = 0.0
    x, end = x_min + 0.2 * width, x_min + 0.8 * width
    while x <= end:
        crossing = sum(1 for w in words if w["x0"] < x < w["x1"])
        if crossing <= allowed_crossings:
            if start is None:
                start = x
            previous = x
        elif start is not None:
            runs.append((start, previous))
            start = None
        x += 1.0
    if start is not None:
        runs.append((start, previous))

    for a, b in sorted(runs, key=lambda r: r[1] - r[0], reverse=True):
        if b - a < MIN_GUTTER_PT:
            break
        centre = (a + b) / 2
        left = sum(1 for w in words if (w["x0"] + w["x1"]) / 2 < centre)
        if min(left, total - left) >= MIN_COLUMN_SHARE * total:
            return centre, b - a
    return None


def _column_text(words: list[dict], gutter_x: float, gutter_w: float) -> str:
    """Read a two-column page: full-width lines in place, columns one after the other."""
    full_width_gap = 0.6 * gutter_w
    blocks: list[str] = []
    left: list[dict] = []
    right: list[dict] = []

    def flush() -> None:
        for column in (left, right):
            if column:
                blocks.append(_words_to_text(column))
        left.clear()
        right.clear()

    for row in _cluster_lines(words):
        row_left = [w for w in row if (w["x0"] + w["x1"]) / 2 < gutter_x]
        row_right = [w for w in row if (w["x0"] + w["x1"]) / 2 >= gutter_x]
        bridges = (
            row_left
            and row_right
            and min(w["x0"] for w in row_right) - max(w["x1"] for w in row_left) < full_width_gap
        )
        if bridges:  # e.g. the name and contact line above the columns
            flush()
            blocks.append(_words_to_text(row))
        else:
            left.extend(row_left)
            right.extend(row_right)
    flush()
    return "\n".join(blocks)


def _page_text(page) -> str:
    words = page.extract_words(x_tolerance=2, y_tolerance=3)
    if len(words) >= 40:
        gutter = _find_gutter(words, page.bbox[0], page.bbox[2])
        if gutter:
            return _column_text(words, *gutter)
    return page.extract_text(x_tolerance=2, y_tolerance=3) or ""


def _plumber_text(data: bytes, max_pages: int) -> tuple[str, int]:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        chunks = [_page_text(page) for page in pdf.pages[:max_pages]]
        return "\n\n".join(chunks), len(pdf.pages)


# --------------------------------------------------------------------------- fallbacks
def _pypdf_text(data: bytes, max_pages: int) -> str:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        reader.decrypt("")
    return "\n\n".join((page.extract_text() or "") for page in reader.pages[:max_pages])


def _ocr_text(data: bytes, max_pages: int) -> str:
    import pytesseract  # imported lazily: OCR is optional and needs the tesseract binary

    chunks = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages[:max_pages]:
            image = page.to_image(resolution=200).original
            chunks.append(pytesseract.image_to_string(image))
    return "\n\n".join(chunks)


# --------------------------------------------------------------------------- public API
def extract_text(
    data: bytes,
    *,
    max_pages: int = 10,
    min_chars: int = 200,
    ocr_enabled: bool = True,
) -> ExtractionResult:
    """Extract text from PDF bytes. Raises ExtractionError with a user-safe message."""
    _check_pdf(data)

    warnings: list[str] = []
    text, total_pages, method = "", 0, "pdfplumber"

    try:
        text, total_pages = _plumber_text(data, max_pages)
    except Exception:  # noqa: BLE001
        logger.warning("pdfplumber failed; trying pypdf", exc_info=True)

    if len(text.strip()) < min_chars:
        try:
            alt = _pypdf_text(data, max_pages)
            if len(alt.strip()) > len(text.strip()):
                text, method = alt, "pypdf"
        except Exception:  # noqa: BLE001
            logger.warning("pypdf failed", exc_info=True)

    if len(text.strip()) < min_chars and ocr_enabled:
        try:
            ocr = _ocr_text(data, max_pages)
            if len(ocr.strip()) > len(text.strip()):
                text, method = ocr, "ocr"
                warnings.append(
                    "This looks like a scanned PDF, so the text was read with OCR and may contain errors."
                )
        except Exception:  # noqa: BLE001
            logger.warning("OCR unavailable or failed", exc_info=True)

    if not total_pages:
        total_pages = len(PdfReader(io.BytesIO(data)).pages)
    if total_pages > max_pages:
        warnings.append(f"Only the first {max_pages} of {total_pages} pages were read.")

    if not text.strip():
        raise ExtractionError(
            "No readable text was found in this PDF. If it is a scan, make sure OCR is enabled "
            "on the server; otherwise export the resume again as a text-based PDF."
        )
    return ExtractionResult(text=text, pages=min(total_pages, max_pages), method=method, warnings=warnings)
