export MAKE_TARGET=$@
export MAKE_SOURCE=$^
CONDOR=$(realpath ./condor) -c
make_detsim:=/home/aiqiang/LINAC/corepmt/make_detsim
make_skg4:=/home/aiqiang/LINAC/corepmt/make_skg4
detsim:=/home/aiqiang/LINAC/corepmt/detsim/skdetsim
skg4:=/home/aiqiang/LINAC/corepmt/skg4

seq:=$(shell seq -w 000001 000013)
.PHONY: all MC 
summaryfile:=$(phase)_linac_runsum.dat
RAWDATA:=$(wildcard $(phasedir)/rfm_lin*.all.root)
all:
	mkdir -p MCskg4/$(phase)
	python3 submit_MC.py --summary $(summaryfile) --phase $(phase) -o MCskg4/$(phase)
MC: $(seq:%=MCskg4/$(phase)/$(run)/%.root)
MCdetsim/$(phase)/$(run)/%.root: MCdetsim/$(phase)/$(run)/g3.%.card
	$(CONDOR) $(detsim) $^ $@ $(run)
MCdetsim/$(phase)/$(run)/g3.%.card: $(make_detsim)/MC.card
	mkdir -p $(dir $@)
	sed -e 's/RUN_NORMAL/$(normalrun)/' -e 's/RAN1/$(shell echo $$RANDOM)/' -e 's/RAN2/$(shell echo $$RANDOM)/' $^ > $@
MCskg4/$(phase)/$(run)/%.root: MCskg4/$(phase)/$(run)/g4.%.mac
	$(CONDOR) "(cd $(skg4) && pwd && source ./G4ROOTsource.sh && ./bin/Linux-g++/SKG4 $(realpath $^) $(shell echo $$RANDOM))"
MCskg4/$(phase)/$(run)/g4.%.mac: linac_$(phase).mac
	mkdir -p $(dir $@)
	sed -e 's/LINAC_RUN/$(run)/' -e's/NORMAL_RUN/$(normalrun)/' -e 's/OUT_FILE/$(subst /,\/,$(CURDIR)/$(dir $@))$*.root/' $^ > $@
linac_sk6.mac: linac.mac.example
	cp $^ $@
linac_sk7.mac: linac.mac.example
	sed -e 's/WaterTransparencyMode 2/WaterTransparencyMode 1/' -e 's/\#\(.*WaterTransparency\) 500/\1 12000/' -e 's/\(IDTBAParameterMode\) 3/\1 1/' -e '107i\/SKG4/Detector/Material/IDTBAParameter 0' $< > $@
# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:

