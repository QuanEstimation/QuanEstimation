FROM python:3.13-slim

LABEL maintainer="QuanEstimation Group https://quanestimation.github.io/group/"
LABEL description="Docker image for QuanEstimation"
LABEL org.opencontainers.image.source="https://github.com/QuanEstimation/QuanEstimation"

ARG BUILD_DATE
ARG VERSION
ARG VCS_REF


LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$VCS_REF

ENV QuanEstimation_INSTALL_JULIA="y" \
    PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
    QuanEstimation_JULIA_PATH="/root/.julia/environments/pyjuliapkg/pyjuliapkg/install/bin/julia" \
    JULIA_NUM_THREADS=auto \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg gcc ca-certificates \
        && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN pip install --no-cache-dir quanestimation

RUN python -c "import quanestimation; print('QuanEstimation installed successfully')"
