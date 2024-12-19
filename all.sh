getmem=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/getmem

for alloc in malloc mmap_anon mmap_file hipMalloc hipHostMalloc hipMallocManaged; do
	$getmem $alloc cpu > $alloc.cpu
	$getmem $alloc gpu > $alloc.gpu
done
