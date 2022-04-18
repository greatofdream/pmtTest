.PHONY: all
export OMP_NUM_THREADS=2

Ex1DataNewDir:=/mnt/neutrino/pmtTest/
# channels:=2 3
wavecut:=0
field:=std
# anaNum
all:  $(anaNum:%=ExResult/%/$(wavecut)ns/charge.h5) $(anaNum:%=ExResult/%/$(wavecut)ns/DN.pdf) $(anaNum:%=ExResult/%/$(wavecut)ns/profile.h5)


define preanalysis

ExResult/$(1)/$(2)ns/h5/preanalysis%.h5: $$(Ex1DataNewDir)$(1)/*$(1)_%.root
	mkdir -p $$(dir $$@)
	time python3 DCRPreAna.py -i $$^ -l $(2) -c $$(channels) -o $$@ > $$@.log 2>&1
ExResult/$(1)/$(2)ns/h5/preanalysis0.h5: $$(wildcard $$(Ex1DataNewDir)$(1)/*$(1).root)
	mkdir -p $$(dir $$@)
	time python3 DCRPreAna.py -i $$^ -l $(2) -c $$(channels) -o $$@ > $$@.log 2>&1
ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5: ExResult/$(1)/$(2)ns/h5/preanalysis0.h5 $$(patsubst %.root,ExResult/$(1)/$(2)ns/h5/preanalysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root))))
	python3 mergeH5.py -i $$^ -o $$@

ExResult/$(1)/$(2)ns/DN.pdf: ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5
	python3 srcuproot3/calcTriggerRate.py -c $$(channels) -i $$^ -o $$@ > $$@.log 2>&1

ExResult/$(1)/$(2)ns/correlation.png: ExResult/$(1)/$(2)ns/preanalysisMerge.h5
	python3 chCorelation.py -i $$^ -c $$(channels)  -o $$(dir $$@) -r $$(Ex1DataNewDir)$(1)/*$(1).root $$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root) > $$@.log 2>&1
ExResult/$(1)/$(2)ns/0$$(field)gr3.pdf: ExResult/$(1)/$(2)ns/preanalysisMerge.h5
	python3 checkByField.py -q 0 -f $$(field) -c $(2) -l 3 -r $$^ -o $$@ -i $$(Ex1DataNewDir)$(1)/*$(1).root $$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root) > $$@.log 2>&1
ExResult/$(1)/$(2)ns/0$$(field)gr4.pdf: ExResult/$(1)/$(2)ns/preanalysisMerge.h5
	python3 checkByField.py -q 0 -f $$(field) -c $(2) -l 4 -r $$^ -o $$@ -i $$(Ex1DataNewDir)$(1)/*$(1).root $$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root) > $$@.log 2>&1
ExResult/$(1)/$(2)ns/0chargegr10.pdf: ExResult/$(1)/$(2)ns/preanalysisMerge.h5
	python3 checkByField.py -q 0 -f minPeakCharge -c $(2) -l 10 -r $$^ -o $$@ -i $$(Ex1DataNewDir)$(1)/*$(1).root $$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root) > $$@.log 2>&1

ExResult/$(1)/$(2)ns/profile.h5: ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5 ExResult/$(1)/$(2)ns/charge.h5
	python3 srcuproot3/waveProfile.py -i $$< --refer $$(word 2,$$^) -c $$(channels)  -o $$@ -r $$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)*.root) > $$@.log 2>&1
ExResult/$(1)/$(2)ns/FP.png: ExResult/$(1)/$(2)ns/preanalysisMerge.h5
	python3 fpCheck.py -i $$^ -c 2 -t 15 -o $$(dir $$@) -r $$(Ex1DataNewDir)$(1)/*$(1).root $$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root) > $$@.log 2>&1


ExResult/$(1)/$(2)ns/0.csv: ExResult/$(1)/$(2)ns/preanalysisMerge.h5
	mkdir -p $$(dir $$@)
	python3 h5Csv.py -c $$(channels) -i $$^ -o $$(dir $$@) > $$@.log 2>&1
ExResult/$(1)/$(2)ns/0.root: ExResult/$(1)/$(2)ns/preanalysisMerge.h5
	mkdir -p $$(dir $$@)
	python3 h5root.py -c $$(channels) -i $$^ -o $$(dir $$@) > $$@.log 2>&1
ExResult/$(1)/$(2)ns/charge0.root: ExResult/$(1)/$(2)ns/0.csv
	root -lbq 'csvroot.C+("$$^","$$@")'
ExResult/$(1)/$(2)ns/charge1.root: ExResult/$(1)/$(2)ns/1.csv
	root -lbq 'csvroot.C+("$$^","$$@")'
ExResult/$(1)/$(2)ns/charge2.root: ExResult/$(1)/$(2)ns/2.csv
	root -lbq 'csvroot.C+("$$^","$$@")'


ExResult/$(1)/$(2)ns/charge.h5: ExResult/$(1)/$(2)ns/h5/preanalysisMerge.h5
	python3 srcuproot3/spePrePlot.py -c $$(channels) -i $$^ -o $$@ > $$@.log 2>&1
ExResult/$(1)/$(2)ns/darkNoise.png: ExResult/$(1)/$(2)ns/preanalysisMerge.h5
	python3 spednPrePlot.py -c $$(channels) -i $$^ -o $$(dir $$@) > $$@.log 2>&1

endef

ExResult/382/19000ns/fit0.png: ExResult/382/19000ns/charge0.root
	root -lbq 'fitChargeRes.C("$^","$@",0,50,150)'
ExResult/382/19000ns/fit1.png: ExResult/382/19000ns/charge1.root
	root -lbq 'fitChargeRes.C("$^","$@",0,65,170)'
ExResult/383/19000ns/fit0.png: ExResult/383/19000ns/charge0.root
	root -lbq 'fitChargeRes.C("$^","$@",0,35,110)'
ExResult/383/19000ns/fit1.png: ExResult/383/19000ns/charge1.root
	root -lbq 'fitChargeRes.C("$^","$@",0,60,160)'

ExResult/386/19000ns/fit0.png: ExResult/386/19000ns/0.root
	root -lbq 'fitChargeRes.C("$^","$@",0,40,170)'
ExResult/386/19000ns/fit1.png: ExResult/386/19000ns/1.root
	root -lbq 'fitChargeRes.C("$^","$@",0,25,120)'
ExResult/386/19000ns/fit2.png: ExResult/386/19000ns/2.root
	root -lbq 'fitChargeRes.C("$^","$@",0,27,137)'

ExResult/387/3500ns/fit0.png: ExResult/387/3500ns/0.root
	root -lbq 'fitChargeRes.C("$^","$@",11,80,60,40,80)'
ExResult/387/3500ns/fit1.png: ExResult/387/3500ns/1.root
	root -lbq 'fitChargeRes.C("$^","$@",40,350,200,100, 250)'
ExResult/387/3500ns/fit2.png: ExResult/387/3500ns/2.root
	root -lbq 'fitChargeRes.C("$^","$@",27,135,120,40, 135)'
ExResult/387/3500ns/fit3.png: ExResult/387/3500ns/3.root
	root -lbq 'fitChargeRes.C("$^","$@",40,350,200,100, 250)'
#echo $$($$(wildcard $$(Ex1DataNewDir)$(1)/*$(1)_*.root):$$(Ex1DataNewDir)$(1)/%.root=ExResult/$(1)/19000ns/preanalysis%.h5)


$(eval $(foreach an,$(anaNum),$(foreach wc,$(wavecut),$(call preanalysis,$(an),$(wc)))))
# >$@.log 2>&1
ExResult/%/19000ns/charge.png: ExResult/%/19000ns/preanalysis.h5
	python3 spePrePlot.py -i $^ -o $(dir $@) > $@.log 2>&1
ExResult/%/19000ns/darkNoise.png: ExResult/%/19000ns/preanalysis.h5
	python3 spednPrePlot.py -i $^ -o $(dir $@) > $@.log 2>&1
ExResult/%/19000ns/0.csv: ExResult/%/19000ns/preanalysis.h5
	mkdir -p $(dir $@)
	python3 h5Csv.py -i $^ -o $(dir $@) > $@.log 2>&1
ExResult/%/19000ns/charge0.root: ExResult/%/19000ns/0.csv
	root -lbq 'csvroot.C+("$^","$@")'
ExResult/%/19000ns/charge1.root: ExResult/%/19000ns/1.csv
	root -lbq 'csvroot.C+("$^","$@")'
ExResult/%/19000ns/fit0.png: ExResult/%/19000ns/charge0.root
	root -lbq 'fitChargeRes.C("$^","$@")'
ExResult/%/19000ns/fit1.png: ExResult/%/19000ns/charge1.root
	root -lbq 'fitChargeRes.C("$^","$@")'

ExResult/%/fine/19000ns/analysis.h5: $(wildcard $(Ex1DataNewDir)*/*.root)
	mkdir -p $(dir $@)
	python3 speAnalysis.py -i $(addprefix $(Ex1DataNewDir)$*/,$(shell ls -v $(Ex1DataNewDir)$*/ --hide=*380_61.root)) -l 19000 -o $@ >$@.log 2>&1
ExResult/%/fine/19000ns/p/charge.png: ExResult/%/fine/19000ns/analysis.h5
	python3 spePlot.py -i $^ -o $(dir $@) > $@.log 2>&1
ExResult/%/fine/19000ns/p1/charge.png: ExResult/%/fine/19000ns/analysis.h5
	mkdir -p $(dir $@)
	python3 spePlot.py -i $^ -o $(dir $@) -p 1 > $@.log 2>&1
ExResult/%/fine/19000ns/p1/0.csv: ExResult/%/fine/19000ns/analysis.h5
	mkdir -p $(dir $@)
	python3 h5Csv.py -i $^ -o $(dir $@) -p 1 > $@.log 2>&1
ExResult/%/fine/19000ns/p1/charge0.root: ExResult/%/fine/19000ns/p1/0.csv
	root -lbq 'csvroot.C+("$^","$@")'
ExResult/%/fine/19000ns/p1/charge1.root: ExResult/%/fine/19000ns/p1/1.csv
	root -lbq 'csvroot.C+("$^","$@")'
ExResult/%/fine/19000ns/p1/fit0.png: ExResult/%/fine/19000ns/p1/charge0.root
	root -lbq 'fitChargeRes.C("$^","$@")'
ExResult/%/fine/19000ns/p1/fit1.png: ExResult/%/fine/19000ns/p1/charge0.root
	root -lbq 'fitChargeRes.C("$^","$@")'
.DELETE_ON_ERROR:
.SECONDARY:
