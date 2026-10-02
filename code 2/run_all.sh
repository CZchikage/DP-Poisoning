#!/usr/bin/env sh
# Reproduces the artifact-covered figures and tables. Runtime is about 15 minutes on one CPU core.
set -e
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
python3 simulations/gaussian_paths.py
python3 simulations/private_ridge.py
python3 certificates.py
python3 downstream.py
python3 tables.py
