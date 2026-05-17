#!/bin/bash
# RQ4 main sweep (Table 6): 4-setting ablation on AMLworld + AMLNet.
# Thin wrapper over scripts/run_main_table.sh that pins DATASETS to the
# cross-dataset subset and uses the 4-encoder Table 6 subset by default.
#
# Output: results/rq4/{amlworld,amlnet}_main_sweep.csv
#
# Example (single GPU):
#   GPU=3 bash scripts/rq4/run_main_sweep.sh
#
# Example (split across GPUs by encoder):
#   GPU=2 ENCODERS="bgrl gbt" TAG=g2 bash scripts/rq4/run_main_sweep.sh &
#   GPU=3 ENCODERS="dgi_bn grace_bn" TAG=g3 bash scripts/rq4/run_main_sweep.sh &
#   wait && bash scripts/merge_main_table.sh
set -e
export DATASETS="amlworld amlnet"
export ENCODERS="${ENCODERS:-bgrl gbt dgi_bn grace_bn}"
exec bash "$(dirname "$0")/../run_main_table.sh" "$@"
