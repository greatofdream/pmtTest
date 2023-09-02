-include mk/run
-include mk/compare

SHELL=bash

.PHONY: all
all: $(RUNS:%=lin/%.pdf) $(ZE:%=compare/%.pdf)

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
