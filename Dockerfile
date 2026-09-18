# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# GridWise — LLM-assisted energy optimizer
#
# Build:
#   docker build -t lokmansharif/gridwise-llm:1.0.0 .
# Run:
#   docker run --rm -p 8000:8000 -e LLM_API_KEY=<key> lokmansharif/gridwise-llm:1.0.0
# Verify:
#   curl http://127.0.0.1:8000/health          -> {"status":"ok"}
#
# Runtime configuration is injected as environment variables; nothing secret is
# baked into any layer. See .env.example for the full list of variable NAMES.
# ---------------------------------------------------------------------------

FROM python:3.12-slim

# Python runtime hygiene: no bytecode, unbuffered logs so container logs stream,
# no pip cache retained in the layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: editing application code then does not invalidate the
# (slow) dependency layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code only. `.env` is excluded via .dockerignore, so the image can
# never carry credentials.
COPY app ./app

# Drop privileges. The service only needs to read its own code and bind a port.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# The documented port. Bind address defaults to all interfaces so the judge
# harness can reach the container.
EXPOSE 8000
ENV HOST=0.0.0.0 \
    PORT=8000

# Container-level readiness probe. urllib is used instead of curl because a slim
# image has no HTTP client installed. A non-zero exit marks the container
# unhealthy; the start period covers the scipy/HiGHS warm-up at boot.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=3)"

# `python -m app.main` honours HOST/PORT (and every other documented variable)
# through app/config.py. The solver warm-up runs during startup, so a healthy
# container already has HiGHS loaded and answers the first request in ~1 s.
CMD ["python", "-m", "app.main"]
