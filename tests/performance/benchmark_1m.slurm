#!/bin/bash
#SBATCH --job-name=sugc-bench-1m
#SBATCH --output=sugc-bench-1m.out
#SBATCH --error=sugc-bench-1m.err
#SBATCH --partition=cosma7-rp,cosma7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --account=dp004

export OMP_NUM_THREADS=16
export RAYON_NUM_THREADS=16

echo "=== Job Started at $(date) ==="
echo "Running benchmark on host: $(hostname)"
echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Using RAYON_NUM_THREADS=$RAYON_NUM_THREADS"

# Execute the python script
/cosma/home/durham/dc-hick2/.local/bin/uv run python /cosma/apps/durham/dc-hick2/SUGC/tests/performance/benchmark_1m.py

echo "=== Job Finished at $(date) ==="
