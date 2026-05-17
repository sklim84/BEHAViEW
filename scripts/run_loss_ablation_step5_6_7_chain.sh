#!/bin/bash
# Chain Steps 5 -> 6 -> 7 after the Step 4 process completes.
set -e
GPU="${GPU:-1}"
STEP4_PID="${STEP4_PID:-}"
echo "[$(date)] === Step 5/6/7 chain (GPU=$GPU) ==="

if [ -n "$STEP4_PID" ]; then
    echo "[$(date)] Waiting for Step 4 PID $STEP4_PID..."
    while kill -0 "$STEP4_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[$(date)] Step 4 finished."
fi

GPU="$GPU" bash scripts/run_loss_ablation_step5.sh
GPU="$GPU" bash scripts/run_loss_ablation_step6.sh
GPU="$GPU" bash scripts/run_loss_ablation_step7.sh
echo "[$(date)] === Step 5/6/7 chain complete ==="
