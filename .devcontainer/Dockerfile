FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Host user/group IDs on dev-gpu01
ARG USERNAME=thorbjoern
ARG USER_UID=1010
ARG USER_GID=1010

# System + CPython build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    openssh-client \
    wget \
    cmake \
    ninja-build \
    pkg-config \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    tk-dev \
    uuid-dev \
    xz-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10.21
RUN cd /tmp \
    && wget https://www.python.org/ftp/python/3.10.21/Python-3.10.21.tgz \
    && tar -xzf Python-3.10.21.tgz \
    && cd Python-3.10.21 \
    && ./configure --prefix=/opt/python/3.10.21 --with-ensurepip=install \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/Python-3.10.21 /tmp/Python-3.10.21.tgz

ENV PATH="/opt/python/3.10.21/bin:${PATH}"

# uv lives inside the container
RUN python3.10 -m pip install --no-cache-dir "uv==0.11.19"

# Match host groups needed for shared storage
RUN groupadd -g 1001 devs \
    && groupadd -g 1003 cache \
    && groupadd -g ${USER_GID} ${USERNAME} \
    && useradd -m -u ${USER_UID} -g ${USER_GID} \
       -G devs,cache \
       -s /bin/bash ${USERNAME}

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=12.0
ENV UV_CACHE_DIR=/srv/cache/uv
ENV UV_LINK_MODE=copy

USER ${USERNAME}
WORKDIR /workspaces/RotGS
