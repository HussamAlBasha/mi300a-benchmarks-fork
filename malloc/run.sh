for n in $(cat n); do numactl -N0 -m0 ./bench $n 100 $1; done | tee $1${HSA_XNACK:+X}.dat
