fault: fault.hip
	hipcc -o $@ -O2 -fopenmp $^ $(EXTRA)

CMD = HIP_VISIBLE_DEVICES=0 numactl -N0 -m0 ./fault -n $*

fault-%.dat:
	HSA_XNACK=1 $(CMD) > $@

nofault-%.dat:
	HSA_XNACK=0 $(CMD) -r > $@

mapped-%.dat:
	HSA_XNACK=0 $(CMD) -m > $@

mappedx-%.dat:
	HSA_XNACK=1 $(CMD) -m > $@

nofaultx-%.dat:
	HSA_XNACK=1 $(CMD) -r > $@

minor-%.dat:
	jSA_XNACK=1 $(CMD) -l > $@

nominor-%.dat:
	HSA_XNACK=0 $(CMD) -lr > $@

nominorx-%.dat:
	HSA_XNACK=1 $(CMD) -lr > $@

touch-%.dat:
	HSA_XNACK=1 $(CMD) -t > $@

touchreg-%.dat:
	HSA_XNACK=0 $(CMD) -tr > $@

touchregx-%.dat:
	HSA_XNACK=1 $(CMD) -tr > $@
