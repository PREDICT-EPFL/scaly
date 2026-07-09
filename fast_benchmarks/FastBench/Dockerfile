# FastBench reproducible image (core + acados).
# Build:  docker build -t fastbench .
# Run:    docker run --rm -v "$PWD/results:/app/results" fastbench \
#             fastbench run --episodes 10
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# core solvers (casadi+ipopt, osqp, piqp) + reporting
RUN pip install --no-cache-dir -e ".[report,dev]"

# acados (embedded backend). Comment out to slim the image.
RUN bash scripts/install_acados.sh
ENV ACADOS_SOURCE_DIR=/app/third_party/acados \
    LD_LIBRARY_PATH=/app/third_party/acados/lib

# pin threads for deterministic timing
CMD ["fastbench", "doctor"]
