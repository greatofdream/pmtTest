.PHONY: all sk6 sk7
all: $(phase1)_$(phase2).csv sk6_summary.csv sk7_summary.csv
sk6:=/disk02/lowe8/sk6/lin/lin.0
sk7:=/disk02/lowe10/sk7/lin/lin.0
compare: $(runs:%=$(state)/%.png)
# echo $(foreach runTuple,$(runTuples),$(word 1,$(subst _, ,$(runTuple))))
$(state)/%.png: 
sk6_summary.csv:
	echo -n > $@
	for i in {86119..86161};do echo $$i; python3 initRuns.py $$i >> $@; done
sk7_summary.csv:
	echo -n > $@
	for i in {91356..91563};do echo $$i; python3 initRuns.py $$i >> $@; done
%_summary.dat: %_summary.csv
	python3 runSummary.py -i $^ -o $@ --phase $* >$@.log
$(phase1)_$(phase2).csv: $(phase1)_summary.csv $(phase2)_summary.csv
	python3 initSetting.py -i $^ --label $(phase1) $(phase2) -o $@
$(phase1)_$(phase2)_ZE.csv: $(phase1)_summary.csv $(phase2)_summary.csv
	python3 initSetting.py -i $^ --label $(phase1) $(phase2) --on Z E --how outer -o $@

define runtuple
result/$(1)_$(2).h5: $$($(phase1))$(1).root $$($(phase2))$(2).root
	mkdir -p $$(dir $$@)
	python3 compare.py $$^ -o $$@

endef

$(eval $(foreach runTuple,$(runTuples),$(call runtuple,$(word 1,$(subst _, ,$(runTuple))),$(word 2,$(subst _, ,$(runTuple))))))
