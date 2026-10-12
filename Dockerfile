FROM python:3.12-slim

# Tesseract is only needed to read scanned (image-only) PDFs.
# Build without it:  docker build --build-arg WITH_OCR=0 -t resume-screener .   (and run with -e OCR_ENABLED=false)
ARG WITH_OCR=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

RUN if [ "$WITH_OCR" = "1" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends tesseract-ocr \
      && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data ./data

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'))"

# Shell form so that ${PORT} (set by Render, Railway, Cloud Run, ...) is expanded.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
