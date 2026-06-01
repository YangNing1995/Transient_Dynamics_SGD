#!/bin/bash
# =====================================================================
# Task #9: Recompute MLP freezing times at different Jaccard thresholds
# from existing endpoint caches.  No GPU needed — runs locally in seconds.
# =====================================================================
set -euo pipefail

CODE_DIR=/appsnew/home/tangc_pkuhpc/lustre1/YN/pytorch/Numerical_experiments
# Directory containing jaccard09/bsX_lrY/endpoints/ caches
ENDPOINT_ROOT=/appsnew/home/tangc_pkuhpc/lustre1/YN/pytorch/freezing_time_training_lr_r1-5_ce/jaccard09
OUTPUT_DIR=/appsnew/home/tangc_pkuhpc/lustre1/YN/pytorch/freezing_time_training_lr_r1-5_ce/threshold_robustness

echo "=== Recomputing freezing times at multiple Jaccard thresholds ==="
echo "Endpoint root: ${ENDPOINT_ROOT}"
echo "Output: ${OUTPUT_DIR}"

python "${CODE_DIR}/recompute_freezing_thresholds.py" \
    --endpoint_root "${ENDPOINT_ROOT}" \
    --thresholds 0.90 0.95 0.99 \
    --output_dir "${OUTPUT_DIR}"

echo "=== Done ==="
