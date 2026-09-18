# syntax=docker/dockerfile:1
# Tennis Technique Analyzer: the browser UI and the whole pipeline, CPU only.
# Built and pushed to ghcr.io by .github/workflows/ci.yml; see docs/DEPLOY.md to run it.

FROM python:3.11-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, in their own layer: code changes do not reinstall PyTorch.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev


FROM python:3.11-slim-bookworm
# ffmpeg for decoding and clips; libGL and glib for OpenCV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app /app
WORKDIR /app

# /data holds everything that must survive an update: config, sessions, uploaded videos,
# downloaded models. Caches go there too, hidden, so models are downloaded only once.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TENNIS_DATA=/data \
    TENNIS_PORT=8731 \
    HOME=/tmp \
    XDG_CACHE_HOME=/data/.cache \
    YOLO_CONFIG_DIR=/data/.cache/ultralytics \
    MPLCONFIGDIR=/data/.cache/matplotlib

# 568 is the "apps" user TrueNAS runs its apps as; any uid works if it can write /data.
USER 568:568
VOLUME /data
EXPOSE 8731
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"TENNIS_PORT\"]}/login', timeout=4)"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
