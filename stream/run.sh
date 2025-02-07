for m in malloc hipMalloc hipHostMalloc hipMallocManaged; do
	#for n in 1 2 4 8 16 32 64 128; do
	for n in 1 3 6 9 12 15 18 21 24; do
		OMP_PROC_BIND=true OMP_NUM_THREADS=$n perf stat -e major-faults,minor-faults,ls_l1_d_tlb_miss.all,dTLB-load-misses,iTLB-load-misses --log-fd 1 -- ./mstream $m
	done > $m.log
done
