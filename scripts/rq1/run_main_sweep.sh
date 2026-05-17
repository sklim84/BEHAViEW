#!/bin/bash
# RQ1 main sweep (Table 3): 4-setting ablation on HOFINET.
# Thin wrapper over scripts/run_main_table.sh that pins
# DATASETS=hofinet and the full 9-encoder set used in Table 3.
#
# Output: results/rq1/main_sweep.csv (after merge_main_table.sh)
#
# Example:
#   GPU=0 ENCODERS="gbt bgrl dgi" TAG=g0 bash scripts/rq1/run_main_sweep.sh &
#   GPU=1 ENCODERS="mvgrl grace dgi_bn" TAG=g1 bash scripts/rq1/run_main_sweep.sh &
#   GPU=2 ENCODERS="mvgrl_bn grace_bn gin" TAG=g2 bash scripts/rq1/run_main_sweep.sh &
#   wait && bash scripts/merge_main_table.sh
set -e
export DATASETS="hofinet"
export ENCODERS="${ENCODERS:-gbt bgrl dgi mvgrl grace dgi_bn mvgrl_bn grace_bn gin}"
exec bash "$(dirname "$0")/../run_main_table.sh" "$@"
