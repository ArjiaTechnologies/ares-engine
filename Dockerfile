FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install .
COPY configs ./configs
COPY DISCLAIMER.md SECURITY.md ./
ENTRYPOINT ["ares"]
CMD ["--help"]

FROM base AS ml
RUN python -m pip install ".[ml]"
