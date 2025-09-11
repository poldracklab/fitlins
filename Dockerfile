FROM ghcr.io/astral-sh/uv:python3.13-alpine AS wheel-builder
RUN apk add --no-cache git
COPY . /app
RUN uv build --wheel /app

FROM continuumio/miniconda3:latest AS conda-env
RUN conda install -c leej3 afni-minimal && \
    conda clean -afy

# Stage 3: Final runtime environment
FROM python:3.13-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=conda-env /opt/conda /opt/conda

WORKDIR /app

COPY pyproject.toml uv.lock .
COPY tools/ ./tools/

COPY --from=wheel-builder /app/dist/*.whl /tmp/

RUN uv sync --locked
RUN uv pip install /tmp/*.whl

ENV PATH="/app/.venv/bin:/opt/conda/bin:$PATH"

ENTRYPOINT ["fitlins"]
