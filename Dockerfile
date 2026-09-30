# syntax=docker/dockerfile:1.7
#
# Layer contract
# --------------
# The large third-party layer is produced in the `deps` stage and is keyed by
# requirements-runtime.txt, the base image, and installer configuration only. No
# workspace metadata reaches it, so bumping an internal package version cannot
# invalidate it. Workspace distributions arrive in the runtime stage as wheels, a
# layer of a few hundred kilobytes.
#
# The runtime stage must not COPY a virtual environment that had workspace packages
# installed into it: that single layer would change on every commit and would undo
# the reuse this file exists to provide.

ARG PYTHON_BASE=python:3.12.13-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv

# --- Stage 1: third-party closure -------------------------------------------
# Keyed by the deterministic projection. Nothing here knows the workspace exists.
FROM ${PYTHON_BASE} AS deps
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
WORKDIR /app
COPY requirements-runtime.txt ./
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    uv venv /app/.venv \
    && uv pip install --python /app/.venv --require-hashes -r requirements-runtime.txt

# --- Stage 2: workspace distributions ---------------------------------------
# Builds wheels only. No dependency resolution and no third-party installation.
FROM ${PYTHON_BASE} AS wheels
ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1
WORKDIR /src
# The build backend is installed from a hash-pinned projection and the build runs with
# --no-build-isolation, so no unhashed backend is fetched from the network mid-build.
COPY requirements-build.txt ./
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    uv venv /build-venv \
    && uv pip install --python /build-venv --require-hashes -r requirements-build.txt
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY packages ./packages
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    VIRTUAL_ENV=/build-venv uv build --all-packages --wheel --no-build-isolation --out-dir /wheels

# --- Stage 3: runtime -------------------------------------------------------
# Contains Python, the production environment, deployment/schema data, generated
# build metadata, and the entrypoint. It contains no uv, no uvx, no package
# manager cache, no tests, no documentation, and no source-control metadata.
FROM ${PYTHON_BASE} AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

# Stable: this layer's content depends only on the `deps` stage. Left root-owned
# and world-readable on purpose. Chowning it would rewrite every file's metadata
# and produce a fresh ~879 MB layer, which is exactly what this file avoids; the
# app user only needs to read and execute it.
COPY --from=deps /app/.venv /app/.venv
COPY --chown=app:app deployments ./deployments
COPY --chown=app:app schemas ./schemas

# Per-commit, and deliberately small. uv, the wheels, the build-only scripts, and
# uv.lock are mounted rather than copied, so none of them lands in the image.
ARG SOURCE_COMMIT=unknown
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=from=wheels,source=/wheels,target=/wheels \
    --mount=type=bind,source=scripts,target=/app/scripts \
    --mount=type=bind,source=uv.lock,target=/app/uv.lock \
    UV_COMPILE_BYTECODE=1 uv pip install --python /app/.venv --no-deps /wheels/*.whl \
    && /app/.venv/bin/python scripts/validate_deployment.py \
    && /app/.venv/bin/python scripts/generate_build_metadata.py --source-commit "$SOURCE_COMMIT"

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "agentic_orchestration.main:app", "--host", "0.0.0.0", "--port", "8000"]
