# Dockerfile for dtype benchmarking
#
# Build: docker build -t dtype-bench .
# Run: docker run --gpus all -v $(pwd)/results:/app/results dtype-bench ./scripts/run_all.sh

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Default command
CMD ["./scripts/run_all.sh"]
