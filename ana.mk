.PHONY: all
all: $(runs:%=$(state)/%.pdf)
# echo $(foreach runTuple,$(runTuples),$(word 1,$(subst _, ,$(runTuple))))
# root -q -l -b 'plot1.C("/disk03/lowe10/sk7/lin/lin.091368.root", "lin.091368.png")'
$(state)/%.pdf: $(directory)%.root
	mkdir -p $(dir $@)
	./wrap_23b root -q -l -b 'plot1.C("$^", "$@")'
