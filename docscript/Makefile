.PHONY:all
datadir:=/mnt/neutrino/pmtTest
all: noisebaseline697_219908_2.pdf noisebaseline697_493375_2.pdf trigger723_101077_4.pdf ser720_ch2.pdf
noisebaseline697_219908_2.pdf: 
	python3 baseline.py --run 697 --eid 219908 --cid 2 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@
noisebaseline697_493375_2.pdf: 
	python3 baseline.py --run 697 --eid 219908 --cid 2 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@
peakcharge697_2.pdf: 
	python3 peakcharge.py --run 697 --cid 2 --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@ 
trigger723_101077_4.pdf:
	python3 triggerwave.py --run 723 --eid 101077 -t 1 --cid 4 --datadir $(datadir) --ana ../ExResult/723/600ns/h5/preanalysisMerge.h5 -o $@ 
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
ser720_ch2_eid302.pdf: $(datadir)/720/Jinping_1ton_Ext_20220529_00000720_302.root ../ExResult/720/600ns/qe/preanalysis302.h5 ../ExResult/720/600ns/chargeSelect.h5
	python3 serfit.py -c 2 3 4 5 --wave $< --ana $(word 2,$^) --summary $(word 3,$^) -o $@ --cid 2 --eid 3140700
ser720_ch2_eid299.pdf: $(datadir)/720/Jinping_1ton_Ext_20220529_00000720_299.root ../ExResult/720/600ns/qe/preanalysis299.h5 ../ExResult/720/600ns/chargeSelect.h5
	python3 serfit.py -c 2 3 4 5 --wave $< --ana $(word 2,$^) --summary $(word 3,$^) -o $@ --cid 2 --eid 3113107
afterpulse720_3406_2.pdf:
	python3 afterpulse.py --run 723 --eid 3406 -t 1 --cid 2 --datadir $(datadir) --ana ../ExResult/723/600ns/qe/preanalysisMerge.h5 -o $@ 