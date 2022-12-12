.PHONY: all laserElastic test
# 根据run的采数配置自动选择处理程序
# example: make anaNum=720
Ex1DataNewDir:=$(shell python3 -c "import config;print(config.databaseDir)")
cpunum:=32
istrigger:=$(shell python3 csvDatabase.py --run $(anaNum) --para istrigger --origincsv $(Ex1DataNewDir)/$(anaNum).csv --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv)
CHANNELS:=$(shell python3 csvDatabase.py --run $(anaNum) --para ch --origincsv $(Ex1DataNewDir)/$(anaNum).csv --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv)
TRIGGERCH:=$(shell python3 csvDatabase.py --run $(anaNum) --para triggerch --origincsv $(Ex1DataNewDir)/$(anaNum).csv --runcsv $(Ex1DataNewDir)/RUNINFO.csv --testcsv $(Ex1DataNewDir)/TESTINFO.csv)

all:
ifeq ($(istrigger),1)
	echo "trigger mode"
	make anaNum=$(anaNum) channels="$(CHANNELS)" Ex1DataNewDir=$(Ex1DataNewDir) triggerch=$(TRIGGERCH) -f Makefiles/Trigger/Makefile -j$(cpunum)
else
	echo "dark noise mode"
	make anaNum=$(anaNum) channels="$(CHANNELS)" Ex1DataNewDir=$(Ex1DataNewDir) -f Makefiles/DN/Makefile -j$(cpunum)
endif

laserElastic:
ifeq ($(istrigger),1)
	echo "trigger mode"
	make anaNum=$(anaNum) channels="$(CHANNELS)" Ex1DataNewDir=$(Ex1DataNewDir) triggerch=$(TRIGGERCH) laserElastic -f Makefiles/Trigger/Makefile -j$(cpunum)
else
	echo "dark noise mode"
	echo "This mode doesn't contain laser signal"
endif
trig:
ifeq ($(istrigger),1)
	echo "trigger mode"
	make anaNum=$(anaNum) channels="$(CHANNELS)" Ex1DataNewDir=$(Ex1DataNewDir) triggerch=$(TRIGGERCH) trig -f Makefiles/Trigger/Makefile -j$(cpunum)
else
	echo "dark noise mode"
	echo "This mode doesn't contain laser signal"
endif
pulse:
ifeq ($(istrigger),1)
	echo "trigger mode"
	make anaNum=$(anaNum) channels="$(CHANNELS)" Ex1DataNewDir=$(Ex1DataNewDir) triggerch=$(TRIGGERCH) pulse -f Makefiles/Trigger/Makefile -j$(cpunum)
else
	echo "dark noise mode"
	echo "This mode doesn't contain laser signal"
endif
test:
ifeq ($(istrigger),1)
	echo "trigger mode"
	echo "anaNum=$(anaNum) channels=$(CHANNELS) Ex1DataNewDir=$(Ex1DataNewDir) triggerch=$(TRIGGERCH) -f Makefiles/Trigger/Makefile -j$(cpunum)"
else
	echo "dark noise mode"
	echo "anaNum=$(anaNum) channels=$(CHANNELS) Ex1DataNewDir=$(Ex1DataNewDir) -f Makefiles/DN/Makefile -j$(cpunum)"
endif
.DELETE_ON_ERROR:
.SECONDARY:
