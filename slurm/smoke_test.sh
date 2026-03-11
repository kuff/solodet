#!/bin/bash
#SBATCH --job-name=solodet-smoke
#SBATCH --output=runs/smoke_%j.out
#SBATCH --error=runs/smoke_%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=batch

set -euo pipefail

cd ~/solodet
mkdir -p runs

CONTAINER=~/solodet/solodet.sif
PYTHON="singularity exec --nv $CONTAINER python"

echo "=== SoloDet Smoke Test: Batch Size Sweep ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""

BATCH_SIZES=(4 8 12 16 20 24)

for BS in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "====================================================="
    echo "Testing batch_size=$BS"
    echo "====================================================="

    # Clean up previous smoke run to avoid conflicts
    rm -rf runs/detect/runs/smoke_test

    if $PYTHON scripts/train.py \
        --model configs/model/yolov8m-p2.yaml \
        --data configs/data/anti_uav.yaml \
        --config configs/train/aicloud_smoke.yaml \
        --pretrained yolov8m.pt \
        --name smoke_test \
        --device 0 \
        --overrides batch=$BS; then

        echo ">>> batch_size=$BS: SUCCESS"

        # Report GPU memory usage
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    else
        echo ">>> batch_size=$BS: FAILED (likely OOM)"
        echo "Stopping sweep."
        break
    fi
done

echo ""
echo "=== Smoke test complete ==="
echo "Finished: $(date)"
