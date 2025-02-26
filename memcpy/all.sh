#!/bin/bash

set -x

for sdma in 0 1; do
    for mem in pageable pinned; do
        HSA_ENABLE_SDMA=$sdma ../hip_bandwidth -mode shmoo -memory $mem > $mem-sdma$sdma.log
    done
done
