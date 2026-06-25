# AMD ROCm PyTorch base image
# See: https://hub.docker.com/r/rocm/pytorch
# Pinned to the latest Ubuntu 22.04 (jammy) build: Python 3.10 (matches the
# MineStudio install in src/wally/collector/AGENTS.md) and the apt package
# names referenced below. The rolling `latest` tag is now Ubuntu 24.04
# (noble), which no longer ships `libgl1-mesa-glx` or `libegl1-mesa`.
FROM rocm/pytorch:rocm7.2.4_ubuntu22.04_py3.10_pytorch_release_2.10.0

RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    gnutls-bin \
    openssh-client \
    libghc-x11-dev \
    gcc-multilib \
    g++-multilib \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    xvfb \
    mesa-utils \
    libegl1-mesa \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    unzip \
    openjdk-8-jdk \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Install MineStudio
RUN pip install --upgrade pip && \
    pip install MineStudio && \
    python -m minestudio.simulator.entry -y

# Replace the ROCm torch from the base image with a CPU-only build.
# librocdxg in WSL2 is broken for RDNA2 (gfx1031), so the AMD torch in this
# image is unusable for compute. CPU torch keeps the gradient / hierarchical
# planners functional (10-50x slower than TheRock GPU on Windows, but
# sufficient for a qualitative "watch the agent plan" loop).
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        --force-reinstall --no-deps torch

WORKDIR /workspace

# Drop into bash
CMD ["bash"]
