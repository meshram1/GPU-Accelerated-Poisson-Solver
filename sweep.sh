#!/usr/bin/env bash
# Per-iteration throughput vs. grid size, CPU vs GPU.
#
# Runs a FIXED iteration count at every size instead of running to convergence:
# Jacobi needs O(N^2) iterations, so a converged run at 4096^2 would take hours
# on the CPU and would conflate "how fast is one sweep" with "how many sweeps
# does Jacobi need". Those are separate questions; this script answers the first.
set -u

ITERS=${ITERS:-2000}
SIZES=${SIZES:-"264 512 1024 2048 4096"}

cd "$(dirname "$0")"

printf '%6s %14s %14s %10s\n' "N" "cpu ms/iter" "gpu ms/iter" "speedup"
printf '%6s %14s %14s %10s\n' "------" "--------------" "--------------" "----------"

for n in $SIZES; do
    out=$(./run_gpu "$n" "$ITERS" 2>&1 | tail -4)
    cpu=$(echo "$out" | grep '^cpu total' | awk '{print $6}')
    gpu=$(echo "$out" | grep '^gpu total' | awk '{print $6}')
    spd=$(echo "$out" | grep '^speedup'   | awk '{print $2}')
    printf '%6s %14s %14s %10s\n' "$n" "${cpu:-ERR}" "${gpu:-ERR}" "${spd:-ERR}"
done
