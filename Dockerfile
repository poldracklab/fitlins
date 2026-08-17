FROM ghcr.io/astral-sh/uv:python3.13-alpine AS wheel-builder
RUN apk add --no-cache git
COPY . /app
RUN uv build --wheel /app

# Stage 3: Final runtime environment
FROM python:3.13-slim AS runtime

# Dependency for 3dBlurToFWHM, 3dFWHMx
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    libexpat1 \
                    libgomp1 \
                    && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install AFNI from Docker container
# Find libraries with `ldd $BINARIES | grep afni`
COPY --link --from=afni/afni_make_build:AFNI_26.2.03 \
    /opt/afni/install/libf2c.so  \
    /opt/afni/install/libmri.so  \
    /usr/local/lib/
COPY --link --from=afni/afni_make_build:AFNI_26.2.03 \
    /opt/afni/install/3dBlurToFWHM \
    /opt/afni/install/3dFWHMx \
    /opt/afni/install/3dPval \
    /opt/afni/install/3dREMLfit \
    /usr/local/bin/

# Changing library paths requires a re-ldconfig
RUN ldconfig

WORKDIR /app

COPY pyproject.toml uv.lock .
COPY tools/ ./tools/
COPY fitlins/ ./fitlins/

COPY --from=wheel-builder /app/dist/*.whl /tmp/

RUN --mount=type=cache,target=/root/.cache/uv uv sync --active --locked --extra test
RUN --mount=type=cache,target=/root/.cache/uv uv pip install /tmp/*.whl

ENV PATH="/app/.venv/bin:/opt/conda/bin:$PATH"

ENTRYPOINT ["fitlins"]
