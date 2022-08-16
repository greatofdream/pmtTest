.PHONY:all
# 根据run的采数配置自动选择处理程序
# example: make anaNum=720
Ex1DataNewDir:=/mnt/neutrino/pmtTest
cpunum:=32

istrigger:=$(shell python3 csvDatabase.py --run $(anaNum) --para istrigger --origincsv $(Ex1DataNewDir)/$(anaNum).csv --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv)
CHANNELS:=$(shell python3 csvDatabase.py --run $(anaNum) --para ch --origincsv $(Ex1DataNewDir)/$(anaNum).csv --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv)
TRIGGERCH:=$(shell python3 csvDatabase.py --run $(anaNum) --para triggerch --origincsv $(Ex1DataNewDir)/$(anaNum).csv --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv)
all:
ifeq ($(istrigger),1)
	echo "trigger mode"
	make anaNum=$(anaNum) channels="$(CHANNELS)" triggerch=$(TRIGGERCH) -f Makefiles/Trigger/Makefile -j$(cpunum)
else
	echo "dark noise mode"
	make anaNum=$(anaNum) channels="$(CHANNELS)" -f Makefiles/DN/Makefile -j$(cpunum)
endif