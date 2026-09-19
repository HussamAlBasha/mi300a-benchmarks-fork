#!/bin/bash -l
#SBATCH --output=/u/halba/mi300a-benchmarks/experiments/allocator_partition_single/logs/job_%x_%j.out
#SBATCH --chdir=/u/halba/mi300a-benchmarks/experiments/allocator_partition_single
#SBATCH --job-name=hip_stream_single
#SBATCH --nodes=1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mail-type=none
#SBATCH --time=00:10:00

set -euo pipefail

experiment_dir=/u/halba/mi300a-benchmarks/experiments/allocator_partition_single
worker="$experiment_dir/worker.sh"

[[ -f "$worker" ]] || { echo "Missing worker: $worker" >&2; exit 1; }

srun --ntasks=1 --cpu-bind=none bash "$worker"
