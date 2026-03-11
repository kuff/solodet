#!/bin/bash
#SBATCH --job-name=solodet-train
#SBATCH --output=runs/train_%j.out
#SBATCH --error=runs/train_%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6-00:00:00
#SBATCH --partition=prioritized

set -euo pipefail

cd ~/solodet
mkdir -p runs

CONTAINER=~/solodet/solodet.sif
PYTHON="singularity exec --nv $CONTAINER python"
RUN_NAME=aicloud_full
WEIGHTS_DIR=runs/detect/runs/${RUN_NAME}/weights

echo "=== SoloDet Full Training ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check for existing checkpoint to auto-resume
if [ -f "${WEIGHTS_DIR}/last.pt" ]; then
    echo ""
    echo "Found existing checkpoint: ${WEIGHTS_DIR}/last.pt"
    echo "Resuming training..."

    $PYTHON scripts/train.py \
        --resume \
        --model "${WEIGHTS_DIR}/last.pt" \
        --data configs/data/anti_uav.yaml \
        --config configs/train/aicloud_full.yaml \
        --name "$RUN_NAME" \
        --device 0
else
    echo ""
    echo "Starting fresh training..."

    $PYTHON scripts/train.py \
        --model configs/model/yolov8m-p2.yaml \
        --data configs/data/anti_uav.yaml \
        --config configs/train/aicloud_full.yaml \
        --pretrained yolov8m.pt \
        --name "$RUN_NAME" \
        --device 0
fi

echo ""
echo "=== Training complete ==="
echo "Finished: $(date)"
