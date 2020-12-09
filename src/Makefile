.PHONY: all
dataDir:=../../ExData/Ex1/2020120*

all: Ex1/364/charge.png Ex1/366/charge.png Ex1/369/1500ns/charge.png Ex1/371/1500ns/charge.png
Ex1/%/preanalysis.h5: 
	mkdir -p $(dir $@)
	python3 spePreAnalysis.py -i $(wildcard $(dataDir)/*$*.root) $(wildcard $(dataDir)/*$*_*.root) -o $@ >$@.log 2>&1
Ex1/%/charge.png: Ex1/%/preanalysis.h5
	python3 spePrePlot.py -i $^ -o $(dir $@) > $@.log 2>&1
Ex1/%/1500ns/preanalysis.h5: 
	mkdir -p $(dir $@)
	python3 spePreAnalysis.py -i $(wildcard $(dataDir)/*$*.root) $(wildcard $(dataDir)/*$*_*.root) -l 1500 -o $@ >$@.log 2>&1
Ex1/%/1500ns/charge.png: Ex1/%/1500ns/preanalysis.h5
	python3 spePrePlot.py -i $^ -o $(dir $@) > $@.log 2>&1
.DELETE_ON_ERROR:
.SECONDARY: