# Slim MineStudio runtime for the wally-dev container.
# The rocm/pytorch base was ~15 GB and we replaced its torch with a CPU
# build on the next line - librocdxg in WSL2 is broken for RDNA2, so
# the AMD stack is unusable. Use plain Ubuntu 22.04 (Python 3.10 native)
# and let `pip install minestudio` pull its own CPU torch.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime deps: Java for the Minecraft engine, LWJGL native libs for
# the MineStudio OpenGL render, xvfb for the headless display.
# -dev variants and Haskell/cross-compile toolchains are dropped.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-venv \
        ca-certificates \
        openjdk-8-jdk \
        libgl1-mesa-glx \
        libglfw3 \
        libegl1-mesa \
        libosmesa6 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libx11-6 \
        libglib2.0-0 \
        xvfb \
        unzip \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install MineStudio (pulls in CPU torch as a dep) and the
# Minecraft engine jar. `python` symlink lets the original
# MineStudio install command work as documented.
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir MineStudio psutil && \
    python -m minestudio.simulator.entry -y && \
    rm -rf /root/.cache/pip

CMD ["bash"]
