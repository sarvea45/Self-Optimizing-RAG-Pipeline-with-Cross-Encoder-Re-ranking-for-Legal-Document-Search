# ── Base Image ─────────────────────────────────────────────────────────────────
# CUDA 12.4 runtime — compatible with your CUDA 12.7 driver (RTX 3050)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ── System Dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ── Working Directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install PyTorch with CUDA 12.4 first (largest dependency) ─────────────────
# Done separately for better Docker layer caching
RUN pip install --no-cache-dir torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Install Python Dependencies ────────────────────────────────────────────────
# Copy requirements first for Docker layer caching
# If only source code changes, this layer is reused (saves ~5 mins)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Download NLTK Data ─────────────────────────────────────────────────────────
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Copy Application Code ──────────────────────────────────────────────────────
COPY . .

# ── Create Required Directories ────────────────────────────────────────────────
RUN mkdir -p data/raw data/processed data/chroma_db results

# ── Expose API Port ────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health Check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start API Server ───────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
