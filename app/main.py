"""FastAPI service. Run locally with:  uvicorn app.main:app --reload"""
from __future__ import annotations

import logging

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import get_settings
from .extractor import ExtractionError
from .scorer import Screener, UnknownRoleError

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("resume_screener")

# Loaded once at start-up: a broken roles.json stops the app immediately instead of failing per request.
screener = Screener.from_settings(settings)

app = FastAPI(
    title="Resume Screener",
    version=__version__,
    description=(
        "Upload a PDF resume and a target role. The service extracts the text, looks for the role's "
        "required skills and keywords, and returns an explainable score and a screening decision. "
        "Resumes are processed in memory and never stored."
    ),
)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if not request.url.path.startswith(("/docs", "/redoc")):  # Swagger UI needs inline scripts
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Content-Security-Policy", "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        )
    return response


def _read_pdf(upload: UploadFile) -> bytes:
    """Read an upload into memory, enforcing type and size limits."""
    name = (upload.filename or "").lower()
    if not name.endswith(".pdf") and upload.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")
    data = upload.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"The file is larger than {settings.max_upload_bytes // (1024 * 1024)} MB.",
        )
    return data


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "version": __version__, "roles": len(screener.roles)}


@app.get("/api/roles", tags=["screening"])
def list_roles() -> list[dict]:
    """Job roles with their default must-have skills, good-to-have skills and keywords."""
    return screener.list_roles()


# These are plain `def` endpoints on purpose: parsing PDFs is CPU-bound, and FastAPI runs
# sync endpoints in a thread pool so they do not block the event loop.
@app.post("/api/screen", tags=["screening"])
def screen_resume(
    file: UploadFile = File(..., description="Resume as a PDF"),
    role: str = Form(..., description="Role id from GET /api/roles"),
    job_description: str = Form("", description="Optional job description text, used for the similarity score"),
) -> dict:
    data = _read_pdf(file)
    try:
        return screener.screen_pdf(data, role, job_description[:5000])
    except UnknownRoleError:
        raise HTTPException(status_code=404, detail=f"Unknown role '{role}'.") from None
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.post("/api/screen/batch", tags=["screening"])
def screen_batch(
    files: list[UploadFile] = File(..., description="Several resumes as PDFs"),
    role: str = Form(..., description="Role id from GET /api/roles"),
) -> dict:
    """Screen many resumes for one role and return them ranked, best first."""
    if len(files) > settings.max_batch_files:
        raise HTTPException(status_code=400, detail=f"Upload at most {settings.max_batch_files} files at a time.")

    loaded: list[tuple[str, bytes]] = []
    rejected: list[dict] = []
    for upload in files:
        try:
            loaded.append((upload.filename or "resume.pdf", _read_pdf(upload)))
        except HTTPException as exc:
            rejected.append({"filename": upload.filename, "error": exc.detail})
    try:
        results = screener.screen_many(loaded, role)
    except UnknownRoleError:
        raise HTTPException(status_code=404, detail=f"Unknown role '{role}'.") from None
    return {"role": role, "count": len(files), "results": results + rejected}
