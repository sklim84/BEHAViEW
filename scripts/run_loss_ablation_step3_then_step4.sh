#!/bin/bash
# Wait for in-flight Step 3 to finish, then dispatch Step 4.
set -e

GPU="${GPU:-1}"
echo "[$(date)] === Step 3 → Step 4 chain (GPU=$GPU) ==="

# Wait for Step 3 process to finish (PID 578613 if still running).
STEP3_PID="${STEP3_PID:-578613}"
while kill -0 "$STEP3_PID" 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Step 3 finished, starting Step 4."
GPU="$GPU" bash scripts/run_loss_ablation_step4.sh
echo "[$(date)] === Chain complete ==="
