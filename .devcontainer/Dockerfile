FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    htop \
    libglib2.0-0t64 \
    libgl1 \
    libx11-6 \
    nvtop \
    wget \
    zip \
    unzip \
    zstd \
 && rm -rf /var/lib/apt/lists/*