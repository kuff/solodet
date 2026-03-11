#!/bin/bash
#SBATCH --job-name=solodet-build
#SBATCH --output=runs/build_%j.out
#SBATCH --error=runs/build_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --partition=batch

set -euo pipefail

cd ~/solodet
mkdir -p runs

echo "=== Building Singularity container via cotainr ==="
echo "Started: $(date)"
echo "Node: $(hostname)"

# Clean up any previous build
rm -f solodet.sif

# Build container with conda env baked in
cotainr build solodet.sif \
    --base-image=docker://nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu22.04 \
    --conda-env=environment.yml \
    --accept-licenses

echo ""
echo "=== Container build complete ==="
echo "Finished: $(date)"
ls -lh solodet.sif
