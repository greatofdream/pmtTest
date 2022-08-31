.PHONY: all
Ex1DataNewDir:=/mnt/neutrino/pmtTest
cpunum:=32
# 初始化PMT信息合并空间
PMTs:=$(shell python3 csvDatabase.py --para pmts --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv)
all: $(PMTs:%=ExPMT/%/config.csv) $(PMTs:%=ExPMT/%/dark.h5) $(PMTs:%=ExPMT/%/laser.h5)

ExPMT/%/config.csv:
	mkdir -p $(dir $@)
	python3 csvDatabase.py --para pmtruns -i $* -o $@ --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv
	touch ExPMT/$*/badrun.csv
ExPMT/%/dark.h5: ExPMT/%/config.csv
	python3 combineDarkAna.py --dir ExResult/{}/0ns/charge.h5 --config $(word 1,$^) --badrun ExPMT/$*/badrun.csv -o $@
ExPMT/%/laser.h5: ExPMT/%/config.csv ExPMT/%/dark.h5
	python3 combineLaserAna.py --dir ExResult/{}/600ns --config $(word 1,$^) --dark $(word 2,$^) --badrun ExPMT/$*/badrun.csv -o $@

.DELETE_ON_ERROR:
.SECONDARY: