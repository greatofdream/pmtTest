.PHONY: all
Ex1DataNewDir:=$(shell python3 -c "import config;print(config.databaseDir)")
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
	python3 combineLaserAna.py --dir ExResult/{}/600ns --config $< --dark ExPMT/$*/dark.h5 --badrun ExPMT/$*/badrun.csv -o $@
	python3 storePMT.py --dark ExPMT/$*/dark.h5 --laser ExPMT/$*/laser.h5 -i ExResult/PMTSummary.csv -o ExPMT/PMTSummary.csv --pmt $*
ExPMT/720-722.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 720 721 722 --ref CR365 > $@.log
ExPMT/761-764.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 761 762 763 764 --ref R5912 > $@.log
ExPMT/775-779.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 775 777 779 --ref CR365 > $@.log
.DELETE_ON_ERROR:
.SECONDARY: