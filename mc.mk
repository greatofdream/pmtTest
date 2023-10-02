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
MCdetsim/$(phase)/$(run)/%.root: MCdetsim/$(phase)/$(run)/g3.%.card
	$(CONDOR) $(detsim) $^ $@ $(run)
MCdetsim/$(phase)/$(run)/g3.%.card: $(make_detsim)/MC.card
	mkdir -p $(dir $@)
	sed -e 's/RUN_NORMAL/$(normalrun)/' -e 's/RAN1/$(shell echo $$RANDOM)/' -e 's/RAN2/$(shell echo $$RANDOM)/' $^ > $@
# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:

