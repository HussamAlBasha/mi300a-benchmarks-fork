#!/bin/bash
set -x
HIP_VISIBLE_DEVICES=0 HSA_XNACK=1 numactl -N0 -m0 ./multichase -F 268435456 -n 10 $*
