export MAKE_TARGET=$@
export MAKE_SOURCE=$^
CONDOR=$(realpath ./condor) -c
.PHONY: all allskg4 RECON RECONMC
summaryfile:=$(phase)_linac_runsum.dat
skg4_mc:=MCskg4/$(phase)
RAWDATA:=$(wildcard $(phasedir)/rfm_lin*.all.root)
MCDATA:=$(wildcard $(phasedir)/*.root)
tentativeFile:=$(if $(tentative),tentative_$(phase).txt,)
tentativeArg:=$(if $(tentative),--tentative,,)
tentativeFlag:=$(if $(tentative),_tentative,,)
linacDat:=./$(phase)_linac_runsum.dat
geDat:=./$(phase)_ge_runsum.dat
all:
	mkdir -p recon/$(phase)
	python3 submit_recon_data.py --summary $(summaryfile) --phase $(phase) -o recon/$(phase) $(tentativeArg)
allskg4:
	mkdir -p reconskg4/$(phase)
	python3 submit_recon_skg4.py --summary $(summaryfile) --phase $(phase) -i $(skg4_mc) -o reconskg4/$(phase) $(tentativeArg)
RECON: $(foreach i,$(RAWDATA),recon/$(phase)/$(run)/$(subst .all,,$(subst rfm_lin$(run).,,$(notdir $i))))
recon/$(phase)/$(run)/%.root: $(phasedir)/rfm_lin$(run).%.all.root
	mkdir -p $(dir $@)
	$(CONDOR) recon_src/lowfit_data$(tentativeFlag) $(run) $@ $^ $(linacDat) $(geDat) $(tentative) $(tentativeFile)
RECONMC: $(foreach i,$(MCDATA),reconskg4/$(phase)/$(run)/$(notdir $i))
reconskg4/$(phase)/$(run)/%.root: $(phasedir)/%.root
	mkdir -p $(dir $@)
	$(CONDOR) recon_src/lowfit_mc$(tentativeFlag) $(run) $@ $^ $(linacDat) $(geDat) $(tentative) $(tentativeFile)

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:

