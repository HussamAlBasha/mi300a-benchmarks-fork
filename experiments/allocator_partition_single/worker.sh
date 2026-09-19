#!/usr/bin/env bash

set -euo pipefail

mode=${MODE:?MODE must be spx, tpx, or cpx}
experiment_dir=/u/halba/mi300a-benchmarks/experiments/allocator_partition_single
repository_dir=/u/halba/mi300a-benchmarks
stream_bin=/u/halba/mi300a-benchmarks/build/rocm-6.3.4/stream
array_elements=33554432
block_size=192

case "$mode" in
    spx) expected_gpu_agents=2; expected_cus=228 ;;
    tpx) expected_gpu_agents=6; expected_cus=76 ;;
    cpx) expected_gpu_agents=12; expected_cus=38 ;;
    *) echo "Invalid MODE: $mode" >&2; exit 2 ;;
esac

module purge
module load gcc/13 openmpi/5.0 rocm/6.3

command -v amd-smi >/dev/null || { echo "amd-smi unavailable" >&2; exit 1; }
command -v rocminfo >/dev/null || { echo "rocminfo unavailable" >&2; exit 1; }
command -v numactl >/dev/null || { echo "numactl unavailable" >&2; exit 1; }
[[ -x "$stream_bin" ]] || { echo "Missing executable: $stream_bin" >&2; exit 1; }

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES HSA_XNACK

job_id=${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}
run_dir="$experiment_dir/runs/${mode}_paper_${job_id}"
provenance_dir="$run_dir/provenance"
mkdir -p "$provenance_dir"

amd-smi static --partition >"$provenance_dir/amd_smi_partition.txt" 2>&1
amd-smi static --asic >"$provenance_dir/amd_smi_asic.txt" 2>&1 || true
amd-smi topology >"$provenance_dir/amd_smi_topology.txt" 2>&1 || true
numactl --hardware >"$provenance_dir/numactl_hardware.txt" 2>&1
hipconfig --full >"$provenance_dir/hipconfig.txt" 2>&1
env HSA_XNACK=0 rocminfo >"$provenance_dir/rocminfo_xnack0.txt" 2>&1
env HSA_XNACK=1 rocminfo >"$provenance_dir/rocminfo_xnack1.txt" 2>&1

partition_file="$provenance_dir/amd_smi_partition.txt"
observed_mode=$(awk -F: '
    /(ACCELERATOR|COMPUTE)_PARTITION:/ {
        gsub(/[[:space:]]/, "", $2)
        value=tolower($2)
        if (value ~ /^(spx|tpx|cpx)$/) print value
    }' "$partition_file" | LC_ALL=C sort -u | paste -sd, -)
memory_mode=$(awk -F: '
    /MEMORY_PARTITION:/ {
        gsub(/[[:space:]]/, "", $2)
        value=toupper($2)
        if (value ~ /^NPS[1248]$/) print value
    }' "$partition_file" | LC_ALL=C sort -u | paste -sd, -)
gpu_count=$(awk '/Device Type:[[:space:]]*GPU/{count++} END{print count+0}' \
    "$provenance_dir/rocminfo_xnack1.txt")
cu_values=$(awk '
    /Device Type:[[:space:]]*GPU/{gpu=1; next}
    gpu && /Compute Unit:/{print $NF; gpu=0}
    ' "$provenance_dir/rocminfo_xnack1.txt" | LC_ALL=C sort -nu | paste -sd, -)

[[ "$observed_mode" == "$mode" ]] || {
    echo "Compute partition mismatch: required=$mode observed=${observed_mode:-unknown}" >&2
    exit 1
}
[[ "$memory_mode" == NPS1 ]] || {
    echo "Memory partition mismatch: required=NPS1 observed=${memory_mode:-unknown}" >&2
    exit 1
}
[[ "$gpu_count" == "$expected_gpu_agents" ]] || {
    echo "GPU-agent mismatch: required=$expected_gpu_agents observed=$gpu_count" >&2
    exit 1
}
[[ "$cu_values" == "$expected_cus" ]] || {
    echo "CU mismatch: required=$expected_cus observed=${cu_values:-unknown}" >&2
    exit 1
}
grep -Eq 'Name:.*xnack-' "$provenance_dir/rocminfo_xnack0.txt" || {
    echo "XNACK=0 validation failed" >&2; exit 1;
}
grep -Eq 'Name:.*xnack\+' "$provenance_dir/rocminfo_xnack1.txt" || {
    echo "XNACK=1 validation failed" >&2; exit 1;
}

cat >"$run_dir/metadata.txt" <<EOF
mode=$mode
logical_device=0
compute_units=$expected_cus
gpu_agents_on_node=$gpu_count
numa_node=0
gpu_initialization=yes
array_elements=$array_elements
array_size_mib=256
arrays=3
block_size=$block_size
iterations=20
units=GB/s
stream_bin=$stream_bin
rocm_module=rocm/6.3
rocm_version=$(hipconfig --version | head -n 1)
hostname=$(hostname)
job_id=$job_id
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_commit=$(git -C "$repository_dir" rev-parse HEAD 2>/dev/null || echo unknown)
EOF

summary_tsv="$run_dir/comparison.tsv"
printf 'allocator\txnack\tcopy_gb_s\tscale_gb_s\tadd_gb_s\ttriad_gb_s\n' >"$summary_tsv"

configurations=(
    malloc:1
    hipMalloc:0 hipMalloc:1
    hipHostMalloc:0 hipHostMalloc:1
    hipMallocManaged:0 hipMallocManaged:1
)

for configuration in "${configurations[@]}"; do
    allocator=${configuration%%:*}
    xnack=${configuration##*:}
    output="$run_dir/${allocator}_xnack${xnack}.out"
    echo "[$(date -u +%FT%TZ)] $mode device 0: $allocator XNACK=$xnack"

    env ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 HSA_XNACK="$xnack" \
        numactl --cpunodebind=0 --membind=0 \
        "$stream_bin" -s -n "$array_elements" -b "$block_size" -v "$allocator" \
        >"$output" 2>&1

    grep -q '^Solution Validates:' "$output" || {
        echo "$allocator XNACK=$xnack did not pass validation; see $output" >&2
        exit 1
    }

    metrics=$(awk '
        $1 == "Copy:"  {copy=$2}
        $1 == "Scale:" {scale=$2}
        $1 == "Add:"   {add=$2}
        $1 == "Triad:" {triad=$2}
        END {
            if (copy == "" || scale == "" || add == "" || triad == "") exit 1
            printf "%s\t%s\t%s\t%s", copy, scale, add, triad
        }' "$output") || {
        echo "Could not parse bandwidth results from $output" >&2
        exit 1
    }
    printf '%s\t%s\t%s\n' "$allocator" "$xnack" "$metrics" >>"$summary_tsv"
done

echo "Results: $run_dir"
