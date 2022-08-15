.PHONY:all
datadir:=/mnt/neutrino/pmtTest
all: noisebaseline697_219908_2.pdf noisebaseline697_493375_2.pdf
noisebaseline697_219908_2.pdf: 
	python3 baseline.py --run 697 --eid 219908 --cid 2 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@ 
noisebaseline697_493375_2.pdf: 
	python3 baseline.py --run 697 --eid 219908 --cid 2 --datadir $(datadir) --ana ../ExResult/697/0ns/h5/preanalysisMerge.h5 -o $@ 