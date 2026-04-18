# Multi-stage build for the RAG assistant. Final image is ~500 MB including
# tesseract (needed for OCR of scanned PDFs). Run with:
#   docker build -t rag .
#   docker run -p 8765:8765 -v $PWD/docs:/app/docs -v $PWD/data:/app/data \
#              --env-file .env rag

# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

WORKDIR /build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --prefix=/install "."

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
      tesseract-ocr-spa \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -m -u 1000 rag
WORKDIR /app
RUN chown rag:rag /app
USER rag

# Default volume mounts — user overrides via -v
VOLUME ["/app/docs", "/app/data"]

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3).status==200 else 1)"

CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8765"]
