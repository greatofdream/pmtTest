.PHONY: all
Ex1DataNewDir:=/mnt/neutrino/pmtTest
cpunum:=32
# 初始化PMT信息合并空间
PMTs:=$(shell python3 csvDatabase.py --para pmts --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv)
all: $(PMTs:%=ExPMT/%/config.csv)

ExPMT/%/config.csv:
	mkdir $(dir $@)
# python3 csvDatabase.py --para pmtruns -i $* -o $@ --origincsv $(Ex1DataNewDir)/$(anaNum).csv --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv
	touch ExPMT/$*/badrun.csv
ExPMT/%/: ExPMT/%/config.csv ExPMT/%/badrun.csv
	python3 combineAna.py --dir ExResult --config $(word 1,$^) --badrun $(word 2,$^)

.DELETE_ON_ERROR:
.SECONDARY: