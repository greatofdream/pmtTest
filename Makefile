-include mk/run

.PHONY: all
all: $(RUNS:%=lin/%.pdf)

mk/run: sk6_summary.csv sk7_summary.csv
	mkdir -p $(@D)
	echo `sh/prepare_raw $^` >> $@

lin/%.pdf: lin/%.root
	mkdir -p $(@D)
	./wrap_23b root -q -l -b 'plot1.C("$^", "$@")'
