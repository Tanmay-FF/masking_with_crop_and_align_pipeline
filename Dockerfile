# Base Image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Maintainer Information
LABEL org.opencontainers.image.authors="Tanmay Thaker <tthaker@gatekeepersystems.com>"
LABEL org.opencontainers.image.contact="tthaker@gatekeepersystems.com"
LABEL org.opencontainers.image.source="https://github.com/facefirst-engineering/ml-docker"
LABEL org.opencontainers.image.vendor="FaceFirst"
LABEL org.opencontainers.image.title="Masked Image generation Pipeline"
LABEL org.opencontainers.image.usage="This image is intended for generating different kinds of augmented images."
LABEL org.opencontainers.image.created="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
LABEL org.opencontainers.image.base="nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04"

# Set environment variables
ENV PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH" \
    NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    NCCL_VERSION="2.15.5-1+cuda12.0" \
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs" \
    DEBIAN_FRONTEND="noninteractive" \
    PYTHONUNBUFFERED=1

# Install system dependencies in a single layer
RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        cmake \
        git \
        python3 \
        python3-dev \
        python3-distutils \
        libjpeg-dev \
        libpng-dev \
        curl \
        gnupg2 \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python and pip
RUN wget -qO get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools


COPY ./requirements.txt /tmp/requirements.txt
# Install all required Python packages
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt


# RUN python3 setup.py build_ext --inplace
WORKDIR /app

# Copy the app source code
COPY . /app
