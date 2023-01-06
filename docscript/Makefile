.PHONY:all
datadir:=/tarski/JNE/JinpingData/Jinping_1ton_Data/pmtTest
all: noisebaseline697_219908_2.pdf noisebaseline697_493375_2.pdf trigger723_101077_4.pdf ser720_ch2.pdf
noisebaseline697_219908_2.pdf: 
	python3 baseline.py --run 697 --eid 219908 --cid 2 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@
noisebaseline697_493375_2.pdf: 
	python3 baseline.py --run 697 --eid 493375 --cid 2 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@
noisebaseline697_39706_5.pdf: 
	python3 baseline.py --run 697 --eid 39706 --cid 5 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@
noisebaseline697_39705_5.pdf:
	python3 baseline.py --run 697 --eid 39705 --cid 5 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@
noisebaseline697_259118_5.pdf: 
	python3 baseline.py --run 697 --eid 259118 --cid 5 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@
noisebaseline697_274295_5.pdf: 
	python3 baseline.py --run 697 --eid 274295 --cid 5 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@
noisebaseline697_298553_5.pdf: 
	python3 baseline.py --run 697 --eid 298553 --cid 5 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@
noisebaseline697_412716_5.pdf: 
	python3 baseline.py --run 697 --eid 412716 --cid 5 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@
peakcharge697_2.pdf: 
	python3 peakcharge.py --run 697 --cid 2 --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@ 
trigger723_101077_4.pdf:
	python3 triggerwave.py --run 723 --eid 101077 -t 1 --cid 4 --datadir $(datadir) --ana ../ExResult/723/600ns/h5/preanalysisMerge.h5 -o $@ 
trigger720_450_2.pdf:
	python3 triggerwave.py --run 720 --eid 450 -t 1 --cid 2 --datadir $(datadir) --ana ../ExResult/720/600ns/h5/preanalysis0.h5 -o $@ 
ser720_ch2.pdf: $(datadir)/720/Jinping_1ton_Ext_20220529_00000720.root ../ExResult/720/600ns/qe/preanalysis0.h5 ../ExResult/720/600ns/chargeSelect.h5
	python3 serfit.py -c 2 3 4 5 --wave $< --ana $(word 2,$^) --summary $(word 3,$^) -o $@ --cid 2
ser720_ch2_eid.pdf: $(datadir)/720/Jinping_1ton_Ext_20220529_00000720.root ../ExResult/720/600ns/qe/preanalysis0.h5 ../ExResult/720/600ns/chargeSelect.h5
	python3 serfit.py -c 2 3 4 5 --wave $< --ana $(word 2,$^) --summary $(word 3,$^) -o $@ --cid 2 --eid 
ser720_ch2_eid1.pdf: $(datadir)/720/Jinping_1ton_Ext_20220529_00000720_1.root ../ExResult/720/600ns/qe/preanalysis1.h5 ../ExResult/720/600ns/chargeSelect.h5
	python3 serfit.py -c 2 3 4 5 --wave $< --ana $(word 2,$^) --summary $(word 3,$^) -o $@ --cid 2 --eid 10928
ser720_ch2_eid56.pdf: $(datadir)/720/Jinping_1ton_Ext_20220529_00000720_56.root ../ExResult/720/600ns/qe/preanalysis56.h5 ../ExResult/720/600ns/chargeSelect.h5
	python3 serfit.py -c 2 3 4 5 --wave $< --ana $(word 2,$^) --summary $(word 3,$^) -o $@ --cid 2 --eid 579644
ser720_ch2_eid284.pdf: $(datadir)/720/Jinping_1ton_Ext_20220529_00000720_284.root ../ExResult/720/600ns/qe/preanalysis284.h5 ../ExResult/720/600ns/chargeSelect.h5
	python3 serfit.py -c 2 3 4 5 --wave $< --ana $(word 2,$^) --summary $(word 3,$^) -o $@ --cid 2 --eid 2955845
ser720_ch2_eid3140700.pdf: $(datadir)/720/Jinping_1ton_Ext_20220529_00000720.root ../ExResult/720/600ns/chargeSelect.h5
	python3 serfit.py -c 2 3 4 5 --wave $< --ana ../ExResult/720/600ns/qe/preanalysis{}.h5 --summary $(word 2,$^) -o $@ --cid 2 --eid 3140700 --run 720
ser720_ch5_eid238481.pdf: $(datadir)/720/Jinping_1ton_Ext_20220529_00000720.root ../ExResult/720/600ns/chargeSelect.h5
	python3 serfit.py -c 2 3 4 5 --wave $< --ana ../ExResult/720/600ns/qe/preanalysis{}.h5 --summary $(word 2,$^) -o $@ --cid 5 --eid 238481 --run 720
ser781_ch5_eid771478.pdf: $(datadir)/781/Jinping_1ton_Ext_20220909_00000781.root  ../ExResult/781/600ns/chargeSelect.h5
	python3 serfit.py -c 2 3 4 5 --wave $< --ana ../ExResult/781/600ns/qe/preanalysis{}.h5 --summary $(word 2,$^) -o $@ --cid 5 --eid 771478 --run 781
afterpulse723_2.pdf:
	python3 afterpulse.py --run 723 -t 1 --cid 2 --datadir $(datadir) --ana ../ExResult/723/600ns/qe/preanalysisMerge.h5 -o $@ 
afterpulse720_2.pdf:
	python3 afterpulse.py --run 720 -t 1 --cid 2 --datadir $(datadir) --ana ../ExResult/720/600ns/qe/preanalysisMerge.h5 -o $@ 
afterpulse761_2.pdf:
	python3 afterpulse.py --run 761 -t 1 --cid 2 --datadir $(datadir) --ana ../ExResult/761/600ns/qe/preanalysisMerge.h5 -o $@ 