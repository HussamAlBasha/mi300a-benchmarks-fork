# Controlled FP64 TRIAD on MI300A partitions

This experiment compares useful TRIAD throughput when each logical device runs alone and when all logical devices on the first MI300A run concurrently. It was created because summing independently selected HIP STREAM peaks does not show whether those peaks occurred during the same interval.

## Build on Viper

From this directory on an allocated MI300A node:

```bash
module purge
module load gcc/13 openmpi/5.0 rocm/6.3 cmake/3.30
set -o pipefail
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build --verbose --parallel 4 2>&1 | tee build/build.log
```

This creates `build/triad_worker`. Rebuild it after changing `triad_worker.hip`.

## Submit

Submit one job for each partition mode:

```bash
mkdir -p logs
for mode in spx tpx cpx; do
    sbatch --mi300-partition="$mode" \
        --job-name="rq5_triad_$mode" \
        --output="$PWD/logs/%x_%j.out" \
        --chdir="$PWD" \
        --export="ALL,MODE=$mode,EXPERIMENT_DIR=$PWD" \
        job.sh
done
```

Each job requests one exclusive Viper node, both GPU resources, and 20 minutes. Results are written to `runs/<mode>_<job-id>/`. Use `python3 run_campaign.py --mode cpx --dry-run` to inspect the CPX schedule without running a benchmark.

## Analyze a run

```bash
python3 analyze.py runs/<mode>_<job-id>
```

The analyzer checks phase completion, numerical validation, trace timing, worker overlap, and isolated/concurrent pairing. It writes CSV summaries and `validation.json` under the run's `analysis/` directory.

## Measurement protocol

- Each worker allocates three FP64 arrays of 33,554,432 elements each. Each array is 256 MiB, giving a fixed 768 MiB working set per worker.
- The worker initializes `a=0`, `b=2`, and `c=3`, then repeatedly computes `a[i] = b[i] + 3.0*c[i]` with a block size of 192 threads.
- Seven allocator/XNACK configurations are measured: `malloc` with XNACK enabled, plus `hipMalloc`, `hipHostMalloc`, and `hipMallocManaged` with XNACK enabled and disabled.
- Each configuration has six traced isolated/concurrent pairs and one untraced pair used to assess tracing sensitivity.
- After 20 warm-up launches, workers continuously issue batches of eight kernels. Complete kernels inside the common two-second interval are counted; 0.2 seconds of work surrounds the interval.
- The coordinator selects the first physical MI300A by PCI address, verifies the active partition and device topology, and assigns a distinct local CPU core to each worker.

Useful throughput is `24 × elements × completed kernels / 2 seconds`, representing two eight-byte reads and one eight-byte write per TRIAD pass. Traced phases require clock-offset uncertainty no greater than 100 microseconds and at least 90% all-worker overlap. Six valid traced pairs are required for a complete configuration mean.

The 768 MiB working set is fixed per worker, so the total package working set increases from SPX to TPX to CPX. Reported useful bytes are an algorithmic count and do not measure physical HBM transactions.

## Source files

| File | Purpose |
| --- | --- |
| `triad_worker.hip` | Allocates and validates arrays, runs TRIAD, and records launch timing. |
| `run_campaign.py` | Builds the schedule and coordinates isolated and concurrent workers. |
| `topology.py` | Verifies partition mode, package membership, devices, and CPU locality. |
| `analyze.py` | Validates saved runs and calculates paired throughput summaries. |
| `job.sh` | Runs the coordinator in a Slurm allocation. |
| `CMakeLists.txt` | Builds `triad_worker` for `gfx942` and links ROCTx. |

Generated builds, runs, logs, and analysis files are ignored by Git.