-include mk/run
-include mk/compare

SHELL=bash

.PHONY: all
all: $(RUNS:%=lin/%.pdf) $(ZE:%=compare/%.pdf) $(RUNS:%=MC/%.root) linac_sk7_runsum.dat

sk6:=$(shell echo {86119..86161})
# 91457 91458 DAQ error; 91561 DAQ stuck
# https://www-sk1.icrr.u-tokyo.ac.jp/sk-local/logbook/SK7/Vol01/logbook.2023.06.12.pdf
# https://www-sk1.icrr.u-tokyo.ac.jp/sk-local/logbook/SK7/Vol01/logbook.2023.06.22.pdf
# 91366 91367 的 descriminator 有问题，导致 trigger 直方图异常
sk7:=$(filter-out 91366 91367 91457 91458 91561,$(shell echo {91356..91563}))
%_summary.csv:
	./initRuns.py $($*) > $@
	sed 's/86157,12/86157,-12/' -i $@ # 86157 的 run summary 写错了
# 用于追加到 /home/sklowe/linac/const/linac_runsum.dat
%_runsum.dat: %_summary.csv
	awk -F, -f awk/precise $^ > $@

mk/run: sk6_summary.csv sk7_summary.csv
	mkdir -p $(@D)
	sh/prepare_raw $^ > $@
mk/compare: sk6_summary.csv sk7_summary.csv
	mkdir -p $(@D)
	./initSetting.py -i $^ -o $@

lin/%.pdf: lin/%.root
	mkdir -p $(@D)
	sh/23b root -q -l -b 'plot1.C("$^", "$@")'

recon_src/lowfit_mc_tentative recon_src/lowfit_data_tentative:
	sh/23b $(MAKE) tentative=1 KAMIOKA_SUKAP64=1 -C $(@D)
recon_src/lowfit_mc recon_src/lowfit_data:
	sh/23b $(MAKE) KAMIOKA_SUKAP64=1 -C $(@D)

SKG4/bin/Linux-g++/SKG4:
	sh/G4 SKG4 ./Make.sh
linac_sk7.mac: linac_sk6.mac
	sed -f sed/SK7 $< > $@
MC/%.root: MC/%.mac SKG4/bin/Linux-g++/SKG4
	[[ -e SKG4/MC ]] || ln -s ../MC SKG4/MC
	sh/G4 SKG4 bin/Linux-g++/SKG4 $< $*

define repeat
$(strip $(shell printf '%.0s $(1)' {1..$(words $(2))}))
endef

.SECONDEXPANSION:
r_sk6=$($*_sk6)
r_sk7=$($*_sk7)
r_files=$(patsubst %,lin/%.root,$(r_sk6) $(r_sk7))
Z_E=$(subst _, ,$*)
Z=$(subst Z,,$(word 1,$(Z_E)))
E=$(subst E,,$(word 2,$(Z_E)))
# SK6 的 run 只有 X=-12
compare/%.pdf: $$(r_files)
	mkdir -p $(@D)
	sh/23b root -l -b -q 'compareM.C("$(r_sk6)","$(r_sk7)","$(call repeat,-12,$(r_sk6))","$(call repeat,-12,$(r_sk7))","lin/","lin/",$(Z),$(E),"$@")'

phase=sk$(phase_$*)
MC/%.mac: linac_$$(phase).mac
	mkdir -p $(dir $@)
	sed -e 's/LINAC_RUN/$*/' -e's/NORMAL_RUN/$(prevn_$*)/' -e 's,OUT_FILE,$(@:.mac=.root),' $^ > $@

.SECONDARY:
.DELETE_ON_ERROR:
