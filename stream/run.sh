for m in malloc hipMalloc hipHostMalloc hipMallocManaged; do
	for n in 1 2 4 8 16 32 64 128; do
		OMP_NUM_THREADS=$n ./mstream $m
	done > $m.log
done
