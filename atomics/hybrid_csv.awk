BEGIN{OFS=","}
{
    for (i=1; i <= NF; i++) {
        print 256 + (i-1)*1024, NR*2, $i
    }
}
