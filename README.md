# Resume Screener

Upload a PDF resume, pick a job role, and get an explainable screening result: a score out of 100, a decision
(**Shortlisted / Needs review / Not shortlisted**), which required skills were found or are missing, and a
marked-up copy of the resume text.

There is no training dataset. Each role has a default profile of **must-have skills**, **good-to-have skills** and
**keywords** stored in `data/roles.json`. The resume is scanned against that profile with deterministic rules plus a
TF-IDF text-similarity signal, so every score can be explained line by line.

Shipped roles: Data Analyst, Data Scientist, Machine Learning Engineer, Python Backend Developer, Frontend Developer.
Add or edit roles by changing the JSON file only; no code changes are needed.

Resumes are processed in memory and are never written to disk.

## How it works

```
PDF -> extractor (pdfplumber -> pypdf -> OCR) -> preprocess (clean, split sections)
    -> parser (skills, contact, experience, education) -> scorer (weights, decision, reasons) -> JSON -> web UI
```

| Score component | Weight | How it is measured |
|---|---|---|
| Must-have skills | 45 | share of required skills found (an "any of" group needs one match) |
| Good-to-have skills | 15 | share found |
| Role keywords | 10 | share found |
| Text similarity | 10 | TF-IDF cosine similarity to the role description (or your pasted job description) |
| Experience | 10 | years found (from date ranges) vs. years the role asks for |
| Education | 10 | degree level found vs. level the role asks for |

A skill that is only listed in the Skills section, and never shows up in experience, projects or certifications,
earns 80% credit. Decision rules: score >= 70 and >= 70% of must-haves gives *Shortlisted*; score >= 50 and >= 50% of
must-haves gives *Needs review*; anything else is *Not shortlisted*. All numbers live in `data/roles.json`.

## Run locally

Python 3.10+ is required. For scanned PDFs, also install the Tesseract OCR engine (optional).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the web UI, or http://127.0.0.1:8000/docs for the interactive API docs.

Tesseract: `sudo apt install tesseract-ocr` (Ubuntu), `brew install tesseract` (macOS), or the UB Mannheim installer
on Windows. Without it, scanned PDFs are rejected with a clear message; text PDFs work as normal.

## API

```bash
curl http://127.0.0.1:8000/api/roles

curl -F "file=@samples/strong_data_scientist.pdf" -F "role=data_scientist" \
     http://127.0.0.1:8000/api/screen

# optional: paste a job description to tune the similarity part of the score
curl -F "file=@resume.pdf" -F "role=data_analyst" -F "job_description=We need SQL and Power BI..." \
     http://127.0.0.1:8000/api/screen

# rank many resumes for one role, best first
curl -F "files=@a.pdf" -F "files=@b.pdf" -F "role=machine_learning_engineer" \
     http://127.0.0.1:8000/api/screen/batch
```

Errors: `404` unknown role, `413` file too large, `415` not a PDF, `422` unreadable/encrypted/empty PDF.

## Add or tune a role

Open `data/roles.json`. A skill written as a list, such as `["Power BI", "Tableau"]`, means "any one of these".
The `aliases` section maps spelling variants to a skill (`sklearn` for `scikit-learn`). Weights must add up to 1.0;
the app refuses to start with an invalid file and tells you why.

```json
"cloud_engineer": {
  "title": "Cloud Engineer",
  "description": "Design, deploy and operate cloud infrastructure...",
  "must_have": ["Linux", "Docker", ["AWS", "GCP", "Azure"], "Git"],
  "good_to_have": ["Kubernetes", "CI/CD"],
  "keywords": ["Infrastructure as Code"],
  "min_experience_years": 1,
  "min_education_level": 2
}
```

`min_education_level`: 1 diploma, 2 bachelor's, 3 master's, 4 PhD.

## Deploy

### Docker (any host)

```bash
docker build -t resume-screener .
docker run -p 8000:8000 resume-screener
# smaller image without OCR:
docker build --build-arg WITH_OCR=0 -t resume-screener .
docker run -p 8000:8000 -e OCR_ENABLED=false resume-screener
```

### Render (free tier)

1. Push this folder to a GitHub repository.
2. On render.com choose **New + > Blueprint** and select the repository. `render.yaml` sets up a Docker web service
   with `/health` as the health check.
3. Open the URL Render gives you. The free tier sleeps when idle, so the first request after a pause is slow.

The `Dockerfile` reads `$PORT`, so it also works on Railway, Google Cloud Run and similar platforms.

### Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_MB` | 5 | largest accepted PDF |
| `MAX_PAGES` | 10 | pages read per resume |
| `MAX_BATCH_FILES` | 20 | files per batch request |
| `OCR_ENABLED` | true | use Tesseract for scanned PDFs |
| `MIN_TEXT_CHARS` | 200 | below this, a PDF is treated as scanned |
| `ROLES_PATH` | data/roles.json | use a different roles file |
| `LOG_LEVEL` | INFO | logging level |

## Test and evaluate

```bash
pip install -r requirements-dev.txt
python -m pytest -q                 # or: python -m unittest discover -s tests

python scripts/make_sample_resumes.py          # regenerate the fictional sample PDFs in samples/
python scripts/evaluate.py samples samples/labels.csv
```

`scripts/evaluate.py` reports accuracy, precision, recall, F1, a confusion matrix, ROC-AUC and a threshold sweep against
a CSV of human labels (`filename,role,label` with `shortlist` or `reject`). The bundled `samples/labels.csv` has only
8 rows and is a smoke test. To judge real-world quality, label 100+ resumes with a recruiter and re-run it, then adjust
weights and thresholds in `roles.json`.

GitHub Actions (`.github/workflows/ci.yml`) runs the tests and a Docker build with a health check on every push.

## Limitations

- Matching is by skill names and aliases, so a skill described without its usual name can be missed. Extend `aliases`.
- Experience is estimated from date ranges and may be off for unusual formats.
- Scanned resumes depend on OCR quality and are slower (about 2-3 seconds vs. under 0.1 second for text PDFs).
- Designs with tables, icons or unusual layouts can scramble text order.
- Scores are a screening aid, not a hiring decision. A person should review every result, especially *Needs review*.
- English resumes only.

## Project layout

```
app/            config, roles, extractor, preprocess, parser, scorer, main (FastAPI) and static/ (web UI)
data/roles.json role profiles, aliases and scoring rules
tests/          unit and API tests
scripts/        sample resume generator and evaluation harness
samples/        fictional test resumes and toy labels
Dockerfile, render.yaml, .github/workflows/ci.yml
```
