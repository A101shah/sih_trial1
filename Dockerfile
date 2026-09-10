# Multi-stage Dockerfile for SatQuery AI Assistant
FROM python:3.10-slim AS base

# System dependencies for OpenCV, GDAL/Rasterio, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        pydantic \
        opencv-python-headless \
        timm \
        einops \
        scikit-learn \
        matplotlib \
        pillow \
        numpy \
        scipy \
        tqdm \
        rasterio \
        shapely \
        transformers \
        peft \
        accelerate

# Copy project files
COPY . /app

# Expose FastAPI & Web Dashboard port
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Start SatQuery AI Server
CMD ["python", "satquery_app.py", "8080"]
