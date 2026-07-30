# ThirtySpokes KOTH validator image — a verify-only daemon (no GPU, no confidential VM).
# Build:  docker compose build           (or: docker build -t thirtyspokes-validator .)
# See docs/OPERATING.md §Validating.
FROM python:3.12-slim

# Native build deps for bittensor / cryptography / dcap-qvl.
# If dcap-qvl has no prebuilt wheel for your platform, add the Rust toolchain before the install step:
#   RUN curl -sSf https://sh.rustup.rs | sh -s -- -y && . "$HOME/.cargo/env"
# docker-CLI is REQUIRED, not optional: the ranked benchmark is code, and `lcb.run_tests` grades it
# by shelling out to `docker run --rm --network none` to execute the submitted program. Without the
# CLI (and the host socket mounted in docker-compose.yml) every proof containing a code task returns
# `grading_unavailable` — MEASURED on testnet 526, where the shipped container could not score the
# ranked benchmark at all and no miner's evidence ever accumulated a single correct answer.
#
# `docker-cli`, NOT `docker.io`: the latter ships only the DAEMON (dockerd, docker-proxy) and lists
# the client as a mere *Recommends*, which `--no-install-recommends` silently drops — so it installs
# cleanly, reports `ii docker.io`, and still leaves no /usr/bin/docker. The daemon is deliberately
# absent: this container talks to the HOST daemon through the mounted socket.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates pkg-config libssl-dev docker-cli \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Validator deps: chain (bittensor + HuggingFace), eval (benchmark datasets + grading), tee (full DCAP).
RUN uv pip install --system --no-cache ".[chain,eval,tee]"

# Wallet is mounted at /root/.bittensor and reign state at /state (see docker-compose.yml).
ENTRYPOINT ["orchestra-koth-validator"]
