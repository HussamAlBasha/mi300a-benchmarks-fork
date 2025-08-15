# MI300A Benchmarks
This repository contains benchmarks for characterizing the memory system and the system software for memory management on AMD's MI300A APU featuring Unified Physical Memory (UPM). The benchmarks are:

* `atomics`: coherence overhead when using atomic operations
* `coherence`: test of HIP coherence mode
* `fault`: page fault overhead
* `hip-stream`: GPU memory bandwidth (modified version of HIP STREAM from AMD, MIT license)
* `malloc`: memory allocation and deallocation
* `measure`: memory usage counter tests
* `memcpy`: hipMemcpy benchmark (unmodified hip-bandwidth from AMD, MIT license)
* `multichase`: memory latency benchmark (modified version of multichase from Google, Apache 2.0 license)
* `stream`: CPU memory bandwidth (modified version of STREAM by John D. McCalpin, custom license)

Unless otherwise noted, the code is copyright 2025 Jacob Wahlgren and is distributed under the MIT license.

If you use these benchmarks in your research, please cite the following paper:

```
@inproceedings{wahlgren2025mi300a,
  title={Dissecting {CPU-GPU} Unified Physical Memory on {AMD MI300A APUs}},
  author={Wahlgren, Jacob and Schieffer, Gabin and Shi, Ruimin and Leon, Edgar and Pearce, Roger and Gokhale, Maya and Peng, Ivy},
  booktitle={2025 IEEE International Symposium on Workload Characterization (IISWC)},
  year={2025},
  organization={IEEE}
}
```

Contact: jacobwah at kth.se
