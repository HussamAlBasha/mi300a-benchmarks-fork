SETUP = HIP_VISIBLE_DEVICES=0 numactl -N0 -m0

all: fault nofault

nofault: EXTRA = -DNOFAULT

fault nofault: fault.hip
	hipcc -o $@ -O2 -fopenmp $^ $(EXTRA)

fault-%.dat: fault
	HSA_XNACK=1 $(SETUP) ./fault $* 4096 > $@

nofault-%.dat: nofault
	HSA_XNACK=0 $(SETUP) ./nofault $* 4096 > $@
