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
Res_$Z_$E.pdf: $(run1:%=$(phase1)%.root) $(run2:%=$(phase2)%.root)
	root -l -b -q 'compareM.C("$(run1)","$(run2)","$(run1x)","$(run2x)","$(phase1)","$(phase2)",$Z,$E,"$@")'

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
