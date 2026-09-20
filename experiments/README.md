# MI300A partition experiments

These three experiments measure FP64 TRIAD bandwidth on the MPCDF Viper cluster across the MI300A SPX, TPX, and CPX compute partition modes.

| Experiment | Purpose |
| --- | --- |
| [`allocator_partition_single`](allocator_partition_single/README.md) | Runs one HIP STREAM process on logical device 0. It compares one 228-CU SPX device, one 76-CU TPX device, and one 38-CU CPX device. |
| [`allocator_partition_concurrent`](allocator_partition_concurrent/README.md) | Runs 1, 3, or 6 HIP STREAM processes concurrently to use every logical partition of the first physical MI300A. It reports the sum of the processes' independently measured peak bandwidths. |
| [`triad_partition_controlled`](triad_partition_controlled/README.md) | Runs a customized TRIAD worker in isolated and concurrent configurations. All workers use the same measurement interval, which enables a controlled comparison of per-partition and aggregate bandwidth. |

The single-process experiment first reproduces the published SPX setup and then applies the same HIP STREAM configuration to TPX and CPX. The concurrent experiment extends this comparison to all partitions of one MI300A. The controlled TRIAD experiment removes the timing limitation of summing independently selected HIP STREAM peaks by measuring concurrent workers over a shared interval.

The two HIP STREAM experiments use the executable built according to [`hip-stream/build.md`](../hip-stream/build.md), expected at:

```text
/u/halba/mi300a-benchmarks/build/rocm-6.3.4/stream
```

The controlled TRIAD experiment builds its own worker. Each experiment README provides its build or submission commands and explains the generated output.
