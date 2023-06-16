.PHONY: all
all: $(state)/compare_$(run1)_$(run2).csv
# echo $(foreach runTuple,$(runTuples),$(word 1,$(subst _, ,$(runTuple))))
$(state)/compare_$(run1)_$(run2).csv: $(phase1)$(run1).root $(phase2)$(run2).root
	root -l -b -q 'compare.C("$<","$(word 2,$^)","$@")'
Res_compare.csv: Res*/compare_*.csv
	echo -n > $@
	for i in $^; do di="$$(dirname $$i)"; name="$$(basename $$i .csv)";di=$${di/Res_/};name=$${name/compare_};echo "$${di//_/,},$${name//_/,},`cat $$i`">> $@; done

