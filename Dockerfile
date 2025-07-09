# Multi-stage build for smaller final image
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies and Python 3.12 in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    python3-pip \
    git \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default and install uv
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.cargo/bin sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create working directory
WORKDIR /app

# Copy requirements file for installation first (for better caching)
COPY requirements.txt .

# Install PyTorch with CUDA 12.8 support and sm120 architecture support
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other requirements from requirements.txt
RUN uv pip install --system -r requirements.txt

# Final stage - runtime image
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Copy Python environment from builder stage
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Create working directory
WORKDIR /app

# Copy the entire application after dependencies are installed
COPY . /app

# Initialize git submodules if they exist
RUN git submodule update --init --recursive || true