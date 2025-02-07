#!/usr/bin/env bash

#https://gist.github.com/parsa/32c7b60e371af1f00fc794fd46b1f98e

set -euo pipefail

[[ -p ctl_fd.fifo ]] && unlink ctl_fd.fifo
mkfifo ctl_fd.fifo
exec {ctl_fd}<>ctl_fd.fifo
echo ctl_fd: $ctl_fd

# NOTE: ACK is optional (--control fd:${ctl_fd},${ctl_fd_ack} to perf-stat)
[[ -p ctl_fd_ack.fifo ]] && unlink ctl_fd_ack.fifo
mkfifo ctl_fd_ack.fifo
exec {ctl_fd_ack}<>ctl_fd_ack.fifo
echo ctrl_fd_ack: $ctl_fd_ack

PERF_CTL_FD=$ctl_fd PERF_ACK_FD=$ctl_fd_ack OMP_NUM_THREADS=24 OMP_PROC_BIND=true numactl -N0 -m0 perf stat --delay=-1 --control fd:${ctl_fd},${ctl_fd_ack} -e major-faults,minor-faults,ls_l1_d_tlb_miss.all,dTLB-load-misses,iTLB-load-misses -- ./mstream $*
