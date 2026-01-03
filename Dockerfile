# syntax=docker/dockerfile:1.6
FROM runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libjpeg-turbo8 \
        libpng16-16 \
        libfreetype6

RUN curl -fsSL https://deno.land/x/install/install.sh | sh -s -- -y \
    && ln -s /root/.deno/bin/deno /usr/local/bin/deno

WORKDIR /workspace/unsloth-playground

COPY pyproject.toml uv.lock ./
RUN python -m pip install --upgrade pip \
    && python -m pip install uv \
    && uv sync --frozen

# Mount or copy the repo into /workspace/unsloth-playground at runtime.
