.PHONY: all
export OMP_NUM_THREADS=2
# Ex1DataNewDir:=/mnt/neutrino/pmtTest
# channels:=2 3
# due to Quantum Efficiency use 
# triggerch:=1
#anaNum:
wavecut:=600
field:=
all: $(anaNum:%=ExResult/%/entries.txt)
all: $(anaNum:%=ExResult/%/$(wavecut)ns/charge.h5)
all: $(anaNum:%=ExResult/%/$(wavecut)ns/trigger.h5)
trig: $(anaNum:%=ExResult/%/$(wavecut)ns/chargeSelect.h5)
laserElastic: $(anaNum:%=ExResult/%/$(wavecut)ns/laserElastic.h5)
ser: $(anaNum:%=ExResult/%/$(wavecut)ns/serMerge.h5)
pulse: $(anaNum:%=ExResult/%/$(wavecut)ns/pulseRatio.h5)
preview: $(anaNum:%=ExResult/%/preview.pdf) $(anaNum:%=ExResult/%/entries.txt) 

define analysis
ExResult/$(1)/preview.pdf: $$(Ex1DataNewDir)/$(1)/*Ext*$(1).root
	mkdir -p $$(dir $$@)
	time python3 trigger/preview.py -i $$^ -c $$(triggerch) $$(channels) -o $$@ > $$@.log 2>&1
ExResult/$(1)/eid%.pdf: $$(Ex1DataNewDir)/$(1)/*Ext*$(1).root
	mkdir -p $$(dir $$@)
	time python3 trigger/preview.py -i $$^ -e $$* -c $$(triggerch) $$(channels) -o $$@ --num ExResult/$(1)/entries.txt
ExResult/$(1)/entries.txt:
	mkdir -p $$(dir $$@)
	time python3 entries.py -f $$(Ex1DataNewDir)/$(1) -o $$@
ExResult/$(1)/$(2)ns/h5/checkFit%.pdf: $$(Ex1DataNewDir)/$(1)/*$(1)_%.root
	mkdir -p $$(dir $$@)
	time python3 checkfit.py -i $$^ -l $(2) -c $$(channels) -t $$(triggerch) -o $$@ > $$@.log 2>&1
ExResult/$(1)/$(2)ns/h5/preanalysis%.h5: $$(Ex1DataNewDir)/$(1)/*$(1)_%.root
	mkdir -p $$(dir $$@)
	time python3 TriggerPreAna.py -i $$^  -l $(2) -c $$(channels) -t $$(triggerch)  -o $$@ > $$@.log 2>&1
ExResult/$(1)/$(2)ns/h5/preanalysis0.h5: $$(Ex1DataNewDir)/$(1)/*$(1).root
	mkdir -p $$(dir $$@)
	time python3 TriggerPreAna.py -i $$^  -l $(2) -c $$(channels) -t $$(triggerch)  -o $$@ > $$@.log 2>&1

ExResult/$(1)/$(2)ns/qe/preanalysis%.h5: $$(Ex1DataNewDir)/$(1)/*$(1)_%.root ExResult/$(1)/$(2)ns/h5/preanalysis%.h5 ExResult/$(1)/$(2)ns/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 TriggerAna.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) --QE > $$@.log 2>&1
ExResult/$(1)/$(2)ns/qe/preanalysis0.h5: $$(Ex1DataNewDir)/$(1)/*$(1).root ExResult/$(1)/$(2)ns/h5/preanalysis0.h5 ExResult/$(1)/$(2)ns/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 TriggerAna.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) --QE > $$@.log 2>&1

ExResult/$(1)/$(2)ns/laserElastic/analysis%.h5: $$(Ex1DataNewDir)/$(1)/*$(1)_%.root ExResult/$(1)/$(2)ns/h5/preanalysis%.h5 ExResult/$(1)/$(2)ns/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 TriggerAna.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) > $$@.log 2>&1
ExResult/$(1)/$(2)ns/laserElastic/analysis0.h5: $$(Ex1DataNewDir)/$(1)/*$(1).root ExResult/$(1)/$(2)ns/h5/preanalysis0.h5 ExResult/$(1)/$(2)ns/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 TriggerAna.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) > $$@.log 2>&1

ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5: ExResult/$(1)/$(2)ns/h5/preanalysis0.h5 $$(patsubst %.root,ExResult/$(1)/$(2)ns/h5/preanalysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)/$(1)/*$(1)_*.root))))
	python3 mergeH5.py -i $$^ -o $$@
ExResult/$(1)/$(2)ns/trigger.h5: ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5
	python TrigInterval.py -i $$^ -o $$@ -c $$(channels) > $$@.log 2>&1

ExResult/$(1)/$(2)ns/qe/preanalysisMerge.h5: ExResult/$(1)/$(2)ns/qe/preanalysis0.h5 $$(patsubst %.root,ExResult/$(1)/$(2)ns/qe/preanalysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)/$(1)/*$(1)_*.root))))
	python3 mergeH5.py -i $$^ -o $$@
ExResult/$(1)/$(2)ns/laserElastic/analysisMerge.h5: ExResult/$(1)/$(2)ns/laserElastic/analysis0.h5 $$(patsubst %.root,ExResult/$(1)/$(2)ns/laserElastic/analysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)/$(1)/*$(1)_*.root))))
	python3 mergeH5.py -i $$^ -o $$@


ExResult/$(1)/$(2)ns/chargeSelect.h5: ExResult/$(1)/$(2)ns/qe/preanalysisMerge.h5 ExResult/$(1)/$(2)ns/trigger.h5
	python3 TriggerSummary.py -c $$(channels) -t $$(word 2,$$^) -i $$< -o $$@ > $$@.log 2>&1
	python3 storeAna.py -i $$@ -o ExResult/TestSummary.csv --run $(1)
ExResult/$(1)/$(2)ns/laserElastic.h5: ExResult/$(1)/$(2)ns/laserElastic/analysisMerge.h5 ExResult/$(1)/$(2)ns/trigger.h5
	echo $$^ $$(word 3,$$^)
	python3 LaserElasticSummary.py -c $$(channels) --interval $$(word 2,$$^) -i $$< -o $$@ > $$@.log 2>&1
# calculate SER
ExResult/$(1)/$(2)ns/ser/ser%.h5: ExResult/$(1)/$(2)ns/qe/preanalysis%.h5 ExResult/$(1)/$(2)ns/chargeSelect.h5 $$(Ex1DataNewDir)/$(1)/*$(1)_%.root
	mkdir -p $$(dir $$@)
	python3 ser.py --ana $$< --summary $$(word 2,$$^) -c $$(channels) -t $$(triggerch) -o $$@ --wave $$(word 3,$$^)
ExResult/$(1)/$(2)ns/ser/ser0.h5: ExResult/$(1)/$(2)ns/qe/preanalysis0.h5 ExResult/$(1)/$(2)ns/chargeSelect.h5 $$(Ex1DataNewDir)/$(1)/*$(1).root
	mkdir -p $$(dir $$@)
	python3 ser.py --ana $$< --summary $$(word 2,$$^) -c $$(channels) -t $$(triggerch) -o $$@ --wave $$(word 3,$$^)
ExResult/$(1)/$(2)ns/serMerge.h5: ExResult/$(1)/$(2)ns/ser/ser0.h5 $$(patsubst %.root,ExResult/$(1)/$(2)ns/ser/ser%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)/$(1)/*$(1)_*.root))))
	python3 ser.py --merge --ana ExResult/$(1)/$(2)ns/qe/preanalysisMerge.h5 --summary ExResult/$(1)/$(2)ns/chargeSelect.h5 --serfiles $$^ -o $$@ -c $$(channels) -t $$(triggerch)
# pre after pulse
ExResult/$(1)/$(2)ns/pulse/preanalysis%.h5: $$(Ex1DataNewDir)/$(1)/*$(1)_%.root ExResult/$(1)/$(2)ns/qe/preanalysis%.h5 ExResult/$(1)/$(2)ns/chargeSelect.h5
	mkdir -p $$(dir $$@)
	python3 TriggerPulseAna.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --summary $$(word 3,$$^) > $$@.log 2>&1
ExResult/$(1)/$(2)ns/pulse/preanalysis0.h5: $$(Ex1DataNewDir)/$(1)/*$(1).root ExResult/$(1)/$(2)ns/qe/preanalysis0.h5 ExResult/$(1)/$(2)ns/chargeSelect.h5
	mkdir -p $$(dir $$@)
	python3 TriggerPulseAna.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --summary $$(word 3,$$^) > $$@.log 2>&1
ExResult/$(1)/$(2)ns/pulseRatio.h5: ExResult/$(1)/$(2)ns/pulse/preanalysis0.h5 $$(patsubst %.root,ExResult/$(1)/$(2)ns/pulse/preanalysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)/$(1)/*$(1)_*.root))))
	python3 TriggerPulseSummary.py -i $$^ -o $$@ -c $$(channels) --ana ExResult/$(1)/$(2)ns/qe/preanalysisMerge.h5 --interval ExResult/$(1)/$(2)ns/trigger.h5
	python3 storeAna.py -i $$@ -o ExResult/TestSummary.csv --run $(1) --pulse

ExResult/$(1)/$(2)ns/charge.h5: ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5
	python3 BasicSummary.py -c $$(channels) -t $$(triggerch) -i $$^ -o $$@ > $$@.log 2>&1
ExResult/$(1)/$(2)ns/ratio.h5: ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5
	cp /srv/greatofdream/JinpingPMTTest/pmtTest/trigger/ExResult/580/400ns/config.json $$(dir $$@)
	cp /srv/greatofdream/JinpingPMTTest/pmtTest/trigger/ExResult/580/400ns/config $$(dir $$@)
	python3 QE.py -c $$(channels) -t $$(triggerch) -i $$^ -o $$@ > $$@.log 2>&1
ExResult/$(1)/$(2)ns/0.root: ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5
	mkdir -p $$(dir $$@)
	python3 h5root.py -c $$(channels) -i $$^ -o $$(dir $$@) > $$@.log 2>&1

endef
$(eval $(foreach an,$(anaNum),$(foreach wc,$(wavecut),$(call analysis,$(an),$(wc)))))

.DELETE_ON_ERROR:
.SECONDARY:
