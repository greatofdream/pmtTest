export MAKE_TARGET=$@
export MAKE_SOURCE=$^
CONDOR=$(realpath ./condor) -c
.PHONY: all compareM compareDataMC compareDataMCone
all: $(state)/compare_$(run1)_$(run2).csv
compareM: $(state).pdf
# echo $(foreach runTuple,$(runTuples),$(word 1,$(subst _, ,$(runTuple))))
$(state)/compare_$(run1)_$(run2).csv: $(phase1)$(run1).root $(phase2)$(run2).root
	root -l -b -q 'compare.C("$<","$(word 2,$^)","$@")'
Res_compare.csv: Res*/compare_*.csv
	echo -n > $@
	for i in $^; do di="$$(dirname $$i)"; name="$$(basename $$i .csv)";di=$${di/Res_/};name=$${name/compare_};echo "$${di//_/,},$${name//_/,},`cat $$i`">> $@; done
dataDir:=recon/$(phase)
mcDir:=reconskg4/$(phase)
summaryfile:=$(phase)_linac_runsum.dat
compareDataMC: 
	python3 submit_compare.py --summary $(summaryfile) --phase $(phase) --datadir $(dataDir) --mcdir $(mcDir) --summary $(summaryfile)
compareDataMCone: compareResult/$(phase)/linac_run$(run).dat
compareResult/$(phase)/linac_run$(run).dat: $(datadir)/$(run)/*.root $(mcdir)/$(run)/*.root
	mkdir -p $(dir $@)
	compare_src/compare $(datadir) $(mcdir) $(dir $@) $(run) $(summaryfile)
compareResult/$(phase)/compare_all.pdf: compareResult/$(phase)/linac_run*.dat
	compare_src/compare_all $(run_b) $(run_e) $(dir $@) $(summaryfile)
compareResult/$(phase)/compare_parameter.pdf: $(sort $(wildcard compareResult/$(phase)/linac_run*.dat))
	python3 compareParameters.py --phase $(phase) -i $(phase)_summary.csv -o $@ -f $^
