# ThirtySpokes KOTH validator image — a verify-only daemon (no GPU, no confidential VM).
# Build:  docker compose build           (or: docker build -t thirtyspokes-validator .)
# See docs/OPERATING.md §Validating.
FROM python:3.12-slim

# Native build deps for bittensor / cryptography / dcap-qvl.
# If dcap-qvl has no prebuilt wheel for your platform, add the Rust toolchain before the install step:
#   RUN curl -sSf https://sh.rustup.rs | sh -s -- -y && . "$HOME/.cargo/env"
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Validator deps: chain (bittensor + HuggingFace), eval (benchmark datasets + grading), tee (full DCAP).
RUN uv pip install --system --no-cache ".[chain,eval,tee]"

# Wallet is mounted at /root/.bittensor and reign state at /state (see docker-compose.yml).
ENTRYPOINT ["orchestra-koth-validator"]
