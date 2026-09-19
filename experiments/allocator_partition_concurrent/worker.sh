#!/usr/bin/env bash

set -euo pipefail

mode=${MODE:?MODE must be spx, tpx, or cpx}
experiment_dir=/u/halba/mi300a-benchmarks/experiments/allocator_partition_concurrent
repository_dir=/u/halba/mi300a-benchmarks
stream_bin=/u/halba/mi300a-benchmarks/build/rocm-6.3.4/stream
array_elements=33554432
block_size=192

case "$mode" in
    spx) concurrent_devices=1; expected_gpu_agents=2; expected_cus=228 ;;
    tpx) concurrent_devices=3; expected_gpu_agents=6; expected_cus=76 ;;
    cpx) concurrent_devices=6; expected_gpu_agents=12; expected_cus=38 ;;
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

device_order=$(seq -s, 0 $((concurrent_devices - 1)))
cat >"$run_dir/metadata.txt" <<EOF
mode=$mode
concurrent_processes=$concurrent_devices
device_order=$device_order
compute_units_per_process=$expected_cus
total_compute_units=228
gpu_agents_on_node=$gpu_count
socket_scope=first_mi300a
numa_node=0
gpu_initialization=yes
array_elements_per_process=$array_elements
array_size_mib_per_process=256
arrays_per_process=3
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

per_device_tsv="$run_dir/per_device.tsv"
aggregate_tsv="$run_dir/comparison.tsv"
printf 'allocator\txnack\tdevice\tcopy_gb_s\tscale_gb_s\tadd_gb_s\ttriad_gb_s\n' \
    >"$per_device_tsv"
printf 'allocator\txnack\tprocesses\tcopy_gb_s\tscale_gb_s\tadd_gb_s\ttriad_gb_s\n' \
    >"$aggregate_tsv"

configurations=(
    malloc:1
    hipMalloc:0 hipMalloc:1
    hipHostMalloc:0 hipHostMalloc:1
    hipMallocManaged:0 hipMallocManaged:1
)
active_pids=()

terminate_children() {
    local pid
    for pid in "${active_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap terminate_children INT TERM EXIT

for configuration in "${configurations[@]}"; do
    allocator=${configuration%%:*}
    xnack=${configuration##*:}
    wave_dir="$run_dir/${allocator}_xnack${xnack}"
    gate="$wave_dir/start.gate"
    mkdir -p "$wave_dir"
    rm -f "$gate"
    active_pids=()
    declare -a devices=()

    echo "[$(date -u +%FT%TZ)] preparing $mode $allocator XNACK=$xnack with $concurrent_devices processes"
    for ((device = 0; device < concurrent_devices; ++device)); do
        device_dir="$wave_dir/device_$(printf '%02d' "$device")"
        output="$device_dir/output.txt"
        mkdir -p "$device_dir"

        (
            while [[ ! -e "$gate" ]]; do
                sleep 0.01
            done
            exec env ROCR_VISIBLE_DEVICES="$device" HSA_XNACK="$xnack" \
                numactl --cpunodebind=0 --membind=0 \
                "$stream_bin" -s -n "$array_elements" -b "$block_size" -v "$allocator"
        ) >"$output" 2>&1 &
        active_pids+=("$!")
        devices+=("$device")
    done

    date -u +%Y-%m-%dT%H:%M:%S.%NZ >"$wave_dir/wave_start.txt"
    : >"$gate"

    wave_failures=0
    for index in "${!active_pids[@]}"; do
        pid=${active_pids[$index]}
        device=${devices[$index]}
        device_dir="$wave_dir/device_$(printf '%02d' "$device")"
        output="$device_dir/output.txt"
        status=0
        wait "$pid" || status=$?

        if [[ "$status" -ne 0 ]] || ! grep -q '^Solution Validates:' "$output"; then
            echo "$allocator XNACK=$xnack device $device failed validation or execution; see $output" >&2
            wave_failures=$((wave_failures + 1))
            continue
        fi

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
            wave_failures=$((wave_failures + 1))
            continue
        }
        printf '%s\t%s\t%s\t%s\n' "$allocator" "$xnack" "$device" "$metrics" \
            >>"$per_device_tsv"
    done
    active_pids=()
    rm -f "$gate"
    date -u +%Y-%m-%dT%H:%M:%S.%NZ >"$wave_dir/wave_end.txt"

    if ((wave_failures > 0)); then
        echo "$wave_failures process(es) failed in $allocator XNACK=$xnack" >&2
        exit 1
    fi

    aggregate=$(awk -F '\t' -v allocator="$allocator" -v xnack="$xnack" '
        NR > 1 && $1 == allocator && $2 == xnack {
            copy += $4; scale += $5; add += $6; triad += $7; count++
        }
        END {
            if (!count) exit 1
            printf "%.4f\t%.4f\t%.4f\t%.4f", copy, scale, add, triad
        }' "$per_device_tsv")
    printf '%s\t%s\t%s\t%s\n' \
        "$allocator" "$xnack" "$concurrent_devices" "$aggregate" >>"$aggregate_tsv"
done

trap - INT TERM EXIT

echo "Results: $run_dir"
