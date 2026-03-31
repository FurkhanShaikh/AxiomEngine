# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Create a non-root user for security.
RUN useradd --create-home --shell /bin/bash axiom

WORKDIR /app

# Copy installed packages from build stage.
COPY --from=builder /install /usr/local

# Copy application source.
COPY src/ ./src/
COPY pyproject.toml .

# Install the package itself (no deps — already installed).
RUN pip install --no-cache-dir --no-deps -e .

USER axiom

EXPOSE 8000

# Uvicorn: single worker per container; scale horizontally via compose/k8s.
CMD ["uvicorn", "axiom_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
