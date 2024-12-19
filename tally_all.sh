tally=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/tally.py

for alloc in malloc mmap_anon mmap_file hipMalloc hipHostMalloc hipMallocManaged; do
    echo $alloc.cpu
    python3 $tally < $alloc.cpu
    echo
    echo
    echo $alloc.gpu
    python3 $tally < $alloc.gpu

    echo
    echo
done
