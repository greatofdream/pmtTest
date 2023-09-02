-include mk/run

mk/run: sk6_summary.csv sk7_summary.csv
	mkdir -p $(@D)
	echo `sh/prepare_raw $^` >> $@
