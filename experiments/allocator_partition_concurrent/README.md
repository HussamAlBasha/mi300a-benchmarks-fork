# Concurrent HIP STREAM on one MI300A

This experiment runs one HIP STREAM process on every logical partition of the first MI300A, keeping the total hardware scope at 228 CUs.

| Mode | Processes | Devices | CUs per process | Total CUs |
| --- | ---: | --- | ---: | ---: |
| SPX | 1 | 0 | 228 | 228 |
| TPX | 3 | 0–2 | 76 | 228 |
| CPX | 6 | 0–5 | 38 | 228 |

The second MI300A remains idle. For each allocator and XNACK configuration, the processes wait at a shared gate and then run concurrently. Every process uses three FP64 arrays of 33,554,432 elements each. Each array is 256 MiB, giving a 768 MiB working set per process. This per-process array size remains fixed across SPX, TPX, and CPX; only the number of concurrent processes changes.

The benchmark uses ROCm 6.3.4, NPS1, NUMA node 0, GPU first touch, block size 192, 20 iterations, and decimal GB/s. Build HIP STREAM first using [the Viper build instructions](../../hip-stream/build.md).

## Submit

From this directory on Viper, submit one job for each partition mode:

```bash
mkdir -p logs runs
for mode in spx tpx cpx; do
    sbatch --job-name="hip_stream_concurrent_${mode}" \
        --mi300-partition="$mode" \
        --export="ALL,MODE=$mode" \
        job.sh
done
```

Each job validates the active partition and records seven concurrent allocator/XNACK waves under `runs/<mode>_paper_<job-id>/`.

## Compare the modes

After all three jobs finish, run:

```bash
python3 compare_results.py --strict --md comparison.md
```

The script selects the latest run for each mode, checks that all configurations and process counts are present, prints the comparison, and creates `comparison.md`. Runs, logs, and the generated report are ignored by Git.

The reported aggregate is the sum of each process's best STREAM bandwidth. The processes start together, but their best iterations may occur at different times, so this is not a cycle-aligned package bandwidth measurement.

## Motivation for the controlled TRIAD experiment

The independently selected STREAM peaks do not prove that all processes achieved those rates during the same interval. This limitation motivated the [controlled TRIAD experiment](../triad_partition_controlled/README.md), which counts completed kernels within one shared measurement interval during isolated and concurrent execution.
