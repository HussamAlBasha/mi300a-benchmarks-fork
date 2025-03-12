fault: fault.hip
	hipcc -o $@ -O2 -fopenmp $^ $(EXTRA)

CMD = HIP_VISIBLE_DEVICES=0 numactl -N0 -m0 ./fault -n $*

fault-%.dat: fault
	HSA_XNACK=1 $(CMD) > $@

nofault-%.dat: fault
	HSA_XNACK=0 $(CMD) -r > $@

minor-%.dat: fault
	HSA_XNACK=1 $(CMD) -l > $@

nominor-%.dat: fault
	HSA_XNACK=1 $(CMD) -lr > $@
