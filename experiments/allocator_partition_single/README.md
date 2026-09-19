# Single-partition HIP STREAM

This experiment runs one HIP STREAM process on logical device 0 to compare the bandwidth available to one logical GPU partition.

| Mode | Device | CUs used |
| --- | ---: | ---: |
| SPX | 0 | 228 |
| TPX | 0 | 76 |
| CPX | 0 | 38 |

The benchmark uses ROCm 6.3.4, NPS1, NUMA node 0, GPU first touch, three 256 MiB FP64 arrays, block size 192, 20 iterations, and decimal GB/s. It runs `malloc` with XNACK enabled and the three HIP allocators with XNACK enabled and disabled. Build HIP STREAM first using [the Viper build instructions](../../hip-stream/build.md).

## Submit

From this directory on Viper, submit one job for each partition mode:

```bash
mkdir -p logs runs
for mode in spx tpx cpx; do
    sbatch --job-name="hip_stream_single_${mode}" \
        --mi300-partition="$mode" \
        --export="ALL,MODE=$mode" \
        job.sh
done
```

Each job validates the active partition and records seven benchmark invocations under `runs/<mode>_paper_<job-id>/`.

## Compare the modes

After all three jobs finish, run:

```bash
python3 compare_results.py --strict --md comparison.md
```

The script selects the latest run for each mode, checks that all configurations are present, prints the comparison, and creates `comparison.md`. Runs, logs, and the generated report are ignored by Git.
