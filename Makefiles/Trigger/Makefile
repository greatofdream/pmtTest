.PHONY: all
export OMP_NUM_THREADS=2


Ex1DataNewDir:=/mnt/neutrino/pmtTest/
# channels:=2 3
# due to Quantum Efficiency use 
# triggerch:=1
#anaNum:
wavecut:=600
field:=
all: $(anaNum:%=ExResult/%/$(wavecut)ns/charge.h5) $(anaNum:%=ExResult/%/$(wavecut)ns/h5/trigger.h5) $(anaNum:%=ExResult/%/$(wavecut)ns/chargeSelect.h5) $(anaNum:%=ExResult/%/$(wavecut)ns/qe/ratio.h5) $(anaNum:%=ExResult/%/$(wavecut)ns/ser.h5)
AP: $(anaNum:%=ExResult/%/$(wavecut)ns/pulse/ratio.pdf)
preview: $(anaNum:%=ExResult/%/preview.pdf) $(anaNum:%=ExResult/%/entries.txt) 

define analysis
ExResult/$(1)/preview.pdf: $$(Ex1DataNewDir)$(1)/*Ext*$(1).root
	mkdir -p $$(dir $$@)
	time python3 trigger/preview.py -i $$^ -c $$(triggerch) $$(channels) -o $$@ > $$@.log 2>&1
ExResult/$(1)/eid%.pdf: $$(Ex1DataNewDir)$(1)/*Ext*$(1).root
	mkdir -p $$(dir $$@)
	time python3 trigger/preview.py -i $$^ -e $$* -c $$(triggerch) $$(channels) -o $$@ --num ExResult/$(1)/entries.txt
ExResult/$(1)/entries.txt:
	mkdir -p $$(dir $$@)
	time python3 entries.py -f $$(Ex1DataNewDir)$(1) -o $$@
ExResult/$(1)/$(2)ns/h5/checkFit%.pdf: $$(Ex1DataNewDir)$(1)/*$(1)_%.root
	mkdir -p $$(dir $$@)
	time python3 checkfit.py -i $$^ -l $(2) -c $$(channels) -t $$(triggerch) -o $$@ > $$@.log 2>&1
ExResult/$(1)/$(2)ns/h5/preanalysis%.h5: $$(Ex1DataNewDir)$(1)/*$(1)_%.root
	mkdir -p $$(dir $$@)
	time python3 TriggerPreAna.py -i $$^  -l $(2) -c $$(channels) -t $$(triggerch)  -o $$@ > $$@.log 2>&1
ExResult/$(1)/$(2)ns/h5/preanalysis0.h5: $$(Ex1DataNewDir)$(1)/*$(1).root
	mkdir -p $$(dir $$@)
	time python3 TriggerPreAna.py -i $$^  -l $(2) -c $$(channels) -t $$(triggerch)  -o $$@ > $$@.log 2>&1

ExResult/$(1)/$(2)ns/pulse/preanalysis%.h5: $$(Ex1DataNewDir)$(1)/*$(1)_%.root ExResult/$(1)/$(2)ns/h5/preanalysis%.h5 ExResult/$(1)/$(2)ns/h5/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 TriggerAna.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) > $$@.log 2>&1
ExResult/$(1)/$(2)ns/pulse/preanalysis0.h5: $$(Ex1DataNewDir)$(1)/*$(1).root ExResult/$(1)/$(2)ns/h5/preanalysis0.h5 ExResult/$(1)/$(2)ns/h5/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 TriggerAna.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) > $$@.log 2>&1

ExResult/$(1)/$(2)ns/qe/preanalysis%.h5: $$(Ex1DataNewDir)$(1)/*$(1)_%.root ExResult/$(1)/$(2)ns/h5/preanalysis%.h5 ExResult/$(1)/$(2)ns/h5/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 TriggerAna.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) --QE > $$@.log 2>&1
ExResult/$(1)/$(2)ns/qe/preanalysis0.h5: $$(Ex1DataNewDir)$(1)/*$(1).root ExResult/$(1)/$(2)ns/h5/preanalysis0.h5 ExResult/$(1)/$(2)ns/h5/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 TriggerAna.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) --QE > $$@.log 2>&1

ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5: ExResult/$(1)/$(2)ns/h5/preanalysis0.h5 $$(patsubst %.root,ExResult/$(1)/$(2)ns/h5/preanalysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root))))
	python3 mergeH5.py -i $$^ -o $$@
ExResult/$(1)/$(2)ns/h5/trigger.h5: ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5
	python trigInterval.py -i $$^ -o $$@ -c $$(channels) > $$@.log 2>&1

ExResult/$(1)/$(2)ns/pulse/preanalysisMerge.h5: ExResult/$(1)/$(2)ns/pulse/preanalysis0.h5 $$(patsubst %.root,ExResult/$(1)/$(2)ns/pulse/preanalysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root))))
	python3 mergeH5.py -i $$^ -o $$@
ExResult/$(1)/$(2)ns/qe/preanalysisMerge.h5: ExResult/$(1)/$(2)ns/qe/preanalysis0.h5 $$(patsubst %.root,ExResult/$(1)/$(2)ns/qe/preanalysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root))))
	python3 mergeH5.py -i $$^ -o $$@
ExResult/$(1)/$(2)ns/chargeSelect.h5: ExResult/$(1)/$(2)ns/qe/preanalysisMerge.h5 ExResult/$(1)/$(2)ns/h5/trigger.h5
	python3 TriggerSummary.py -c $$(channels) -t $$(word 2,$$^) -i $$< -o $$@ > $$@.log 2>&1
ExResult/$(1)/$(2)ns/ser.h5: ExResult/$(1)/$(2)ns/qe/preanalysisMerge.h5 ExResult/$(1)/$(2)ns/chargeSelect.h5
	python3 ser.py --ana $$< --summary $$(word 2,$$^) -c $$(channels) -t $$(triggerch) -o $$@ --wave $$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)*.root) > $$@.log 2>&1

ExResult/$(1)/$(2)ns/pulse/ratio.pdf: ExResult/$(1)/$(2)ns/pulse/preanalysisMerge.h5
	python3 trigger/pulseRatio.py -i $$^ -o $$@ > $$@.log
ExResult/$(1)/$(2)ns/qe/ratio.h5: ExResult/$(1)/$(2)ns/qe/preanalysisMerge.h5 ExResult/$(1)/$(2)ns/h5/trigger.h5
	python3 trigger/QERatio.py -i $$< -o $$@ --interval $$(word 2,$$^) -c $$(channels) > $$@.log

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

ExResult/401/3500ns/fit2.png: ExResult/401/3500ns/2.root
	root -lbq 'fitChargeRes.C("$^","$@",27,135,120,40, 135)'

ExResult/409/400ns/fit1.png: ExResult/409/400ns/1.root
	root -lbq '../srcuproot3/fitChargeRes.C("$^","$@",4,85,30,20, 50)'
ExResult/410/400ns/fit3.png: ExResult/410/400ns/3.root
	root -lbq '../srcuproot3/fitChargeRes.C("$^","$@",6,170,103,90, 120)'
ExResult/411/400ns/fit3.png: ExResult/411/400ns/3.root
	root -lbq '../srcuproot3/fitChargeRes.C("$^","$@",6,170,120,110, 135)'
ExResult/650/400ns/pulse/ratio.log: ExResult/650/400ns/pulse/ratio.pdf.log
	python3 pulseRatioWOnoise.py -i $< -o $@ --ref ../srcuproot3/ExResult/651/peak.pdf.log
ExResult/654/400ns/pulse/ratio.log: ExResult/654/400ns/pulse/ratio.pdf.log
	python3 pulseRatioWOnoise.py -i $< -o $@ --ref ../srcuproot3/ExResult/653/peak.pdf.log
ExResult/657/400ns/pulse/ratio.log: ExResult/657/400ns/pulse/ratio.pdf.log
	python3 pulseRatioWOnoise.py -i $< -o $@ --ref ../srcuproot3/ExResult/656/peak.pdf.log
.DELETE_ON_ERROR:
.SECONDARY:
