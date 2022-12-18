.PHONY: all
Ex1DataNewDir:=$(shell python3 -c "import config;print(config.databaseDir)")
cpunum:=32
# 初始化PMT信息合并空间
PMTs:=$(shell python3 csvDatabase.py --para pmts --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv)
all: $(PMTs:%=ExPMT/%/config.csv) $(PMTs:%=ExPMT/%/dark.h5) $(PMTs:%=ExPMT/%/laser.h5)

ExPMT/%/config.csv:
	mkdir -p $(dir $@)
	python3 csvDatabase.py --para pmtruns -i $* -o $@ --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv
ExPMT/%/badrun.csv:
	touch $@
ExPMT/%/dark.h5: ExPMT/%/config.csv ExPMT/%/badrun.csv
	python3 combineDarkAna.py --dir ExResult/{}/0ns/charge.h5 --config $(word 1,$^) --badrun $(word 2,$^) -o $@
ExPMT/%/laser.h5: ExPMT/%/config.csv ExPMT/%/dark.h5
	python3 combineLaserAna.py --dir ExResult/{}/600ns --config $< --dark ExPMT/$*/dark.h5 --badrun ExPMT/$*/badrun.csv -o $@ > $@.log
	python3 storePMT.py --dark ExPMT/$*/dark.h5 --laser ExPMT/$*/laser.h5 -i ExResult/PMTSummary.csv -o ExPMT/PMTSummary.csv --pmt $*
ExPMT/PMTSummary.pdf: ExPMT/PMTSummary.csv
	python3 darkLaserCompare.py -i $^ -o $@ > $@.log
ExPMT/OLS/720-722.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 720 721 722 --ref CR365 > $@.log
ExPMT/OLS/761-764.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 761 762 763 764 --ref R5912 > $@.log
ExPMT/OLS/775_777_779_784.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 775 777 779 784 --ref CR365 > $@.log
ExPMT/OLS/786_788_790_792.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 786 788 790 792 --ref CR365 > $@.log
ExPMT/OLS/794_796_798_800.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 794 796 798 800 --ref CR365 > $@.log
ExPMT/GLM/720-722.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 720 721 722 --ref CR365 --glm > $@.log
ExPMT/GLM/761-764.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 761 762 763 764 --ref R5912 --glm > $@.log
ExPMT/GLM/775_777_779_784.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 775 777 779 784 --ref CR365 --glm > $@.log
	python3 storePDE.py -i $@ --csv ExResult/TestSummary.csv --calibcsv ExPMT/Calibration.csv --runs 775 777 779 784
ExPMT/GLM/786_788_790_792.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 786 788 790 792 --ref CR365 --glm > $@.log
	python3 storePDE.py -i $@ --csv ExResult/TestSummary.csv --calibcsv ExPMT/Calibration.csv --runs 786 788 790 792
ExPMT/GLM/794_796_798_800.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 794 796 798 800 --ref CR365 --glm > $@.log
	python3 storePDE.py -i $@ --csv ExResult/TestSummary.csv --calibcsv ExPMT/Calibration.csv --runs 794 796 798 800
ExPMT/GLM/802.h5:
	python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o $@ --runs 802 --ref CR365 > $@.log
.DELETE_ON_ERROR:
.SECONDARY: