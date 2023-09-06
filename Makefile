-include mk/run
-include mk/compare

SHELL=bash

.PHONY: all
all: $(RUNS:%=lin/%.pdf) $(ZE:%=compare/%.pdf)

sk6:=$(shell echo {86119..86161})
# 91457 91458 DAQ error; 91561 DAQ stuck
# https://www-sk1.icrr.u-tokyo.ac.jp/sk-local/logbook/SK7/Vol01/logbook.2023.06.12.pdf
# https://www-sk1.icrr.u-tokyo.ac.jp/sk-local/logbook/SK7/Vol01/logbook.2023.06.22.pdf
# 91366 91367 的 descriminator 有问题，导致 trigger 直方图异常
sk7:=$(filter-out 91366 91367 91457 91458 91561,$(shell echo {91356..91563}))
%_summary.csv:
	parallel -j1 ./initRuns.py {} ::: $($*) > $@
	sed 's/86157,12/86157,-12/' -i $@ # 86157 的 run summary 写错了

mk/run: sk6_summary.csv sk7_summary.csv
	mkdir -p $(@D)
	echo `sh/prepare_raw $^` >> $@
mk/compare: sk6_summary.csv sk7_summary.csv
	mkdir -p $(@D)
	./initSetting.py -i $^ -o $@

lin/%.pdf: lin/%.root
	mkdir -p $(@D)
	./wrap_23b root -q -l -b 'plot1.C("$^", "$@")'

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
	./wrap_23b root -l -b -q 'compareM.C("$(r_sk6)","$(r_sk7)","$(call repeat,-12,$(r_sk6))","$(call repeat,-12,$(r_sk7))","lin/","lin/",$(Z),$(E),"$@")'

.SECONDARY:
.DELETE_ON_ERROR:
