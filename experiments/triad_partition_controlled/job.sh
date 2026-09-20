#!/bin/bash -l
#SBATCH --job-name=rq5_triad
#SBATCH --nodes=1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --mail-type=none
set -euo pipefail

experiment_dir=${EXPERIMENT_DIR:?set EXPERIMENT_DIR when submitting the job}

module purge
module load gcc/13 openmpi/5.0 rocm/6.3

# Full allocation CPU affinity; the coordinator selects distinct local physical cores.
srun --ntasks=1 --cpu-bind=none python3 "$experiment_dir/run_campaign.py" \
    --mode "${MODE:?}" --binary "${TRIAD_BIN:-$experiment_dir/build/triad_worker}" --execute
