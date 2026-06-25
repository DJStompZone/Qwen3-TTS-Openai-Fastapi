# syntax=docker/dockerfile:1.7

# Qwen3-TTS OpenAI-Compatible API Server
# CUDA-only image for NVIDIA GPUs
# Target GPU: NVIDIA RTX 4060 / Ada Lovelace / CUDA arch 8.9

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04
ARG PYTORCH_CUDA=cu128
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}
ARG BUILDER_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

# =============================================================================
# Stage 1: Runtime base
# =============================================================================
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV CUDA_HOME=/usr/local/cuda

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    htop \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Builder with CUDA compiler for flash-attn
# =============================================================================
FROM ${BUILDER_IMAGE} AS builder

ARG PYTORCH_CUDA

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=1
ENV NVCC_THREADS=1
ENV TORCH_CUDA_ARCH_LIST=8.9

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    ninja-build \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /build

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir \
    "torch>=2.0.0" \
    "torchaudio>=2.0.0" \
    --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"

RUN pip install --no-cache-dir \
    "transformers>=4.40.0" \
    "accelerate>=1.0.0" \
    librosa \
    soundfile \
    pydub \
    numpy \
    scipy \
    einops \
    onnxruntime-gpu \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    python-multipart \
    "pydantic>=2.0.0" \
    inflect \
    aiofiles \
    ninja \
    packaging \
    psutil

RUN mkdir -p /wheelhouse \
    && MAX_JOBS="${MAX_JOBS}" NVCC_THREADS="${NVCC_THREADS}" pip wheel --no-cache-dir --no-build-isolation --wheel-dir /wheelhouse flash-attn \
    && pip install --no-cache-dir /wheelhouse/flash_attn*.whl

RUN python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available during build:", torch.cuda.is_available())

import flash_attn
import flash_attn_2_cuda

print("flash-attn import: OK")
print("flash_attn_2_cuda import: OK")
PY

# =============================================================================
# Export built wheels for flash-attn
# =============================================================================
FROM scratch AS wheelhouse

COPY --from=builder /wheelhouse /

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM base AS production

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

COPY . .

RUN pip install --no-cache-dir -e .

RUN python - <<'PY'
import torch
import flash_attn
import flash_attn_2_cuda

print("production torch:", torch.__version__)
print("production torch CUDA:", torch.version.cuda)
print("production flash-attn import: OK")
print("production flash_attn_2_cuda import: OK")
PY

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /models /tmp/numba_cache /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /models /tmp/numba_cache /home/appuser/.cache

USER appuser

ENV HOST=0.0.0.0
ENV PORT=8880
ENV WORKERS=1
ENV PYTHONPATH=/app
ENV TTS_BACKEND=official
ENV TTS_MODEL_NAME=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

CMD ["python", "-m", "api.main"]