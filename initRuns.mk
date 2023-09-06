.PHONY: all sk6 sk7
all: $(phase1)_$(phase2).csv
compare: $(runs:%=$(state)/%.png)
# echo $(foreach runTuple,$(runTuples),$(word 1,$(subst _, ,$(runTuple))))
$(state)/%.png: 
%_linac_runsum.dat: %_summary.csv
	python3 runSummary.py -i $^ -o $@ --phase $* >$@.log
%_ge_runsum.dat: %_summary.csv
	python3 runSummary.py -i $^ -o $@ --phase $* --ge >$@.log

define runtuple
result/$(1)_$(2).h5: $$($(phase1))$(1).root $$($(phase2))$(2).root
	mkdir -p $$(dir $$@)
	python3 compare.py $$^ -o $$@

endef

$(eval $(foreach runTuple,$(runTuples),$(call runtuple,$(word 1,$(subst _, ,$(runTuple))),$(word 2,$(subst _, ,$(runTuple))))))
