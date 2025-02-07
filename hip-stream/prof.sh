for xnack in 0 1; do
    for alloc in malloc hipMalloc hipHostMalloc hipMallocManaged malloc+register hipHostMallocNonCoherent hipMallocManagedCpu; do
        HSA_XNACK=$xnack numactl -N0 -m0 rocprofv3 -i counters.txt -o "${alloc}${xnack}" -d tlb -- ./stream -s "$alloc"
    done
done
