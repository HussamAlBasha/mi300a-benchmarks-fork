for k in $(seq 10 32); do
    n=$((2**$k))
    echo -n $n,
    ./run.sh -m $n $*
done
