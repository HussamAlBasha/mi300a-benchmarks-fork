# Build HIP STREAM on the Viper GPU cluster

These instructions are specific to the MPCDF Viper cluster and its AMD MI300A GPU nodes. The retained experiments used the `gfx942` target and the Viper `rocm/6.3` module, which resolved to ROCm 6.3.4. The referenced paper used ROCm 6.3.1, which was not available through the Viper module environment used for these experiments.

## Load the Viper modules

```bash
module purge
module load gcc/13 openmpi/5.0 rocm/6.3 cmake/3.30
```

Open MPI is required because the HIP STREAM CMake configuration links its MPI wrapper.

## Configure and build

The experiment scripts expect the repository at `/u/halba/mi300a-benchmarks` and the executable at `build/rocm-6.3.4/stream`. From any directory on Viper, run:

```bash
repository_dir=/u/halba/mi300a-benchmarks
source_dir="$repository_dir/hip-stream"
build_dir="$repository_dir/build/rocm-6.3.4"

cmake --fresh \
    -S "$source_dir" \
    -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES=gfx942

cmake --build "$build_dir" --parallel "$(nproc)"
```

`cmake --fresh` replaces an existing CMake cache, so the build directory does not need to be deleted first. The resulting executable is `/u/halba/mi300a-benchmarks/build/rocm-6.3.4/stream`.

## Check the executable

The help command can run without a GPU:

```bash
"$build_dir/stream" -h
```

Run the benchmark inside a Slurm allocation on a Viper GPU node. An allocator argument is required:

```bash
srun "$build_dir/stream" -v hipMalloc
```

The `-v` option validates the computed arrays after the benchmark.

## Run the experiments

The partition experiments are under `experiments/allocator_partition_single` and `experiments/allocator_partition_concurrent`. Their READMEs contain the submission commands and measurement protocols. Generated runs and Slurm logs remain outside Git.
