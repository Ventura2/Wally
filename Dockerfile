# AMD ROCm PyTorch base image
# See: https://hub.docker.com/r/rocm/pytorch
FROM rocm/pytorch:latest

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

WORKDIR /workspace

# Drop into bash
CMD ["bash"]
