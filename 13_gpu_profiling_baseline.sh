#!/bin/bash
set -euo pipefail

# 13 — GPU profiling baseline for VaVAM-B M=1 on DRIVE AGX Thor
#
# This profiles the EXISTING 12j GPU-resident implementation.
# It does not rebuild engines and does not use the per-step D2H diagnostic.
#
# Expected working directory:
#   ~/vblkdev2/VaVAM_Thor/thor_eval
#
# Usage:
#   bash 13_gpu_profiling_baseline.sh
#
# Output:
#   nsys_vavam_B_M1_baseline.nsys-rep
#   nsys_vavam_B_M1_baseline.sqlite (if export succeeds)
#   nsys_vavam_B_M1_baseline_stats.txt

NSYS="${NSYS:-/usr/local/bin/nsys}"
OUT="${OUT:-nsys_vavam_B_M1_baseline}"
PYTHON="${PYTHON:-python3}"
APP="${APP:-12j_vavam_B_evaluate_ego_trajectory.py}"

if [[ ! -x "$NSYS" ]]; then
    echo "[ERROR] nsys not found/executable: $NSYS"
    exit 1
fi

if [[ ! -f "$APP" ]]; then
    echo "[ERROR] Cannot find $APP in $(pwd)"
    exit 1
fi

echo "=============================================================="
echo "13 — GPU profiling baseline"
echo "=============================================================="
echo "Nsight Systems : $NSYS"
"$NSYS" --version
echo "Application     : $APP"
echo "Output          : $OUT"
echo

echo "[1/2] Running Nsight Systems..."
echo "NOTE: profiling adds some overhead; this run is for kernel/timeline"
echo "      attribution, not for replacing the 12j latency benchmark."
echo

"$NSYS" profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --stats=true \
    --force-overwrite=true \
    -o "$OUT" \
    "$PYTHON" "$APP" --index 0 --seed 0

echo
echo "[2/2] Generating post-run stats..."
if "$NSYS" stats "$OUT.nsys-rep" > "${OUT}_stats.txt" 2>&1; then
    echo "Stats saved: ${OUT}_stats.txt"
else
    echo "[WARN] nsys stats returned non-zero."
    echo "The .nsys-rep may still be valid. Check the captured report."
fi

echo
echo "=============================================================="
echo "DONE"
echo "=============================================================="
echo "Report : $OUT.nsys-rep"
echo "Stats  : ${OUT}_stats.txt"
echo
echo "Next: copy/paste the following sections from ${OUT}_stats.txt:"
echo "  - CUDA API Statistics"
echo "  - CUDA Kernel Statistics"
echo "  - CUDA Memory Operation Statistics"
echo
echo "Then we will map kernels/time to:"
echo "  Prefill / Action Expert / Euler / other CUDA overhead."
