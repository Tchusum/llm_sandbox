FROM ghcr.io/astral-sh/uv as uv_stage

FROM mcr.microsoft.com/devcontainers/base

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/* 

# Install uv
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=uv_stage /uv /uvx /bin/

CMD ["sleep", "infinity"]

