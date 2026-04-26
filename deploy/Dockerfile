# Single Dockerfile shared by all three services. Compose picks the entry
# point per container (worker / master / lb).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS deps required by faiss + sentence-transformers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-services.txt ./

# Install the CPU-only torch wheel first. The default torch wheel from PyPI
# bundles ~2 GB of NVIDIA CUDA libraries (cublas, cudnn, etc.) that we never
# use because the simulated LLM backend runs on CPU. Installing torch from
# PyTorch's CPU index up-front means sentence-transformers (a transitive dep)
# sees torch already satisfied and won't pull the GPU wheel.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt -r requirements-services.txt

COPY . .

# Compose overrides CMD per service. This default is just a sanity check.
CMD ["python", "-c", "print('Override the CMD in docker-compose.yml')"]
