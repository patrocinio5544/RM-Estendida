FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/MedNeXt

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    graphviz \
    libgdcm-tools \
    libgdcm-dev \
    python3-gdcm \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch validado no WSL: torch 2.1.0 + CUDA 11.8
RUN pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements_docker.txt /workspace/MedNeXt/requirements_docker.txt

RUN pip install -r requirements_docker.txt

COPY . /workspace/MedNeXt

RUN pip install -e .

CMD ["/bin/bash"]
