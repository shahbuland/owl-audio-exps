# Multi-stage build for smaller final image
FROM nvcr.io/nvidia/pytorch:24.12-py3 AS builder

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

# Install uv using the existing conda environment
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/conda/bin sh
ENV PATH="/opt/conda/bin:${PATH}"

# Create working directory
WORKDIR /app

# Copy requirements file for installation first (for better caching)
COPY requirements.txt .

# PyTorch is already installed in the NGC base image, skip PyTorch installation

# Install other requirements from requirements.txt using conda's pip
RUN /opt/conda/bin/uv pip install --system --break-system-packages -r requirements.txt

# Final stage - runtime image
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install minimal runtime dependencies including OpenGL libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python is already configured in the NGC base image

# Copy Python packages from builder stage
COPY --from=builder /opt/conda /opt/conda

# Create working directory
WORKDIR /app

# Copy the entire application after dependencies are installed
COPY . /app

# Initialize git submodules if they exist and checkout specified branch
RUN git submodule update --init --recursive || true && \
    git submodule foreach --recursive 'git checkout $branch || git checkout $sha1 || true'

# Copy the environment file (Do this last)
COPY .env .