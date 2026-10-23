# motion-toolbox runtime image
# Base: CUDA 12.4 + cuDNN (matches the torch==2.6.0 cu124 wheels pinned in constraints.txt).
# Python 3.11 (.python-version / requires-python >= 3.11) is installed by uv, which is also
# the package manager the README uses (`uv sync`).
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# uv binary (pinned) from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Headless matplotlib.
    MPLBACKEND=Agg \
    # uv: copy (not hardlink) across the layer filesystem; keep the interpreter and the
    # virtualenv *outside* /workspace so the host repo mounted there does not hide them.
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# --- System dependencies ------------------------------------------------------
# git: `mhr` extra and some scripts fetch from git.
# ffmpeg/libgl1/libglib2.0: needed by opencv-python, imageio-ffmpeg (viz extra).
# wget/curl/unzip: used by scripts/*.sh download helpers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Managed Python 3.11 (same version as .python-version).
RUN uv python install 3.11

WORKDIR /workspace

# --- Python dependencies ------------------------------------------------------
# Only the files needed for an editable install are copied; the full repo is mounted at
# /workspace at runtime (see run.sh), which is also where the editable install points.
COPY pyproject.toml README.md .python-version ./
COPY src ./src
COPY docker-shell/constraints.txt /tmp/constraints.txt

# Core deps + `viz` and `semantic-eval` extras + the `dev` dependency group (pytest, ruff,
# mypy). `ml` (JAX), `mhr` (needs a Pixi/pymomentum build, see docs/MHR_SETUP_GUIDE.md) and
# `report` (WeasyPrint system libs) are left out of the image.
RUN uv venv /opt/venv --python 3.11 \
    && uv pip install \
        --constraint /tmp/constraints.txt \
        --group dev \
        -e ".[viz,semantic-eval]"

# Quick sanity check that the core stack imported cleanly and CUDA build is present.
RUN python -c "import torch, motion_toolbox, cv2, spacy; \
print('torch', torch.__version__, '| cuda build', torch.version.cuda)"

CMD ["/bin/bash"]
