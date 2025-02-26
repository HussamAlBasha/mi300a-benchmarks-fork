log = pageable-sdma1.log pinned-sdma1.log pageable-sdma0.log pinned-sdma0.log
dats = $(patsubst %.log,%.h2d.dat,$(log)) $(patsubst %.log,%.d2h.dat,$(log)) $(patsubst %.log,%.d2d.dat,$(log))

all.csv: $(dats)
	echo $(dats) | tr " " , > $@
	paste -d, $^ >> $@

%.h2d.dat: %.log
	grep -F 'Bandwidth Host to Device' $< | sed -e  's/^.*\[//' -e 's/, /\n/g' | tr -d ' ]' > $@

%.d2h.dat: %.log
	grep -F 'Bandwidth Device to Host' $< | sed -e  's/^.*\[//' -e 's/, /\n/g' | tr -d ' ]' > $@

%.d2d.dat: %.log
	grep -F 'Bandwidth Device to Device' $< | sed -e  's/^.*\[//' -e 's/, /\n/g' | tr -d ' ]' > $@
