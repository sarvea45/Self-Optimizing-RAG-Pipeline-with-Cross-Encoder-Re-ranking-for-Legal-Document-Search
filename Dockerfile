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

# ── Install PyTorch with CUDA 12.4 ────────────────────────────────────────────
# Separate layer for caching. Uses --no-cache-dir to avoid hash issues.
RUN pip install --no-cache-dir --timeout 120 \
    torch==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# ── Install Python Dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

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