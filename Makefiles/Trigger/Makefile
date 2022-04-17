.PHONY: all
export OMP_NUM_THREADS=2


Ex1DataNewDir:=/mnt/neutrino/pmtTest/
channels:=2 3
# due to Quantum Efficiency use 
#triggerch:=0
triggerch:=1
#anaNum:=401 402 403 404 405 406 407 408 409 410 411
#anaNum:=507 508 509 511
anaNum:=512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 529 530 531 532 533 534 535 536 538 539 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 557 558 559 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 632 633 634 635 643 649 650 651 654 655 657 659 661 662 664 666 668 670 672 674 676 678
pauNum:=643 650
wavecut:=400
field:=
#Pause:$(pauNum:%=Ex1/%/pauseMergy.h5)
QE: $(anaNum:%=Ex1/%/400ns/qe.h5)
define analysis
Ex1/$(1)/preview.pdf: $$(Ex1DataNewDir)$(1)/*Ext*$(1).root
	mkdir -p $$(dir $$@)
	time python3 preview.py -i $$^ -c $$(triggerch) $$(channels) -o $$@ > $$@.log 2>&1
Ex1/$(1)/$(2)ns/h5/checkFit%.pdf: $$(Ex1DataNewDir)$(1)/*$(1)_%.root
	mkdir -p $$(dir $$@)
	time python3 checkfit.py -i $$^ -l $(2) -c $$(channels) -t $$(triggerch) -o $$@ > $$@.log 2>&1
Ex1/$(1)/$(2)ns/h5/preanalysis%.h5: $$(Ex1DataNewDir)$(1)/*$(1)_%.root
	mkdir -p $$(dir $$@)
	time python3 triggerPreAnalysis.py -i $$^  -l $(2) -c $$(channels) -t $$(triggerch)  -o $$@ > $$@.log 2>&1
Ex1/$(1)/$(2)ns/h5/preanalysis0.h5: $$(Ex1DataNewDir)$(1)/*$(1).root
	mkdir -p $$(dir $$@)
	time python3 triggerPreAnalysis.py -i $$^  -l $(2) -c $$(channels) -t $$(triggerch)  -o $$@ > $$@.log 2>&1
Ex1/$(1)/$(2)ns/pulse/preanalysis%.h5: $$(Ex1DataNewDir)$(1)/*$(1)_%.root Ex1/$(1)/$(2)ns/h5/preanalysis%.h5 Ex1/$(1)/$(2)ns/h5/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 triggerPulse.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) > $$@.log 2>&1
Ex1/$(1)/$(2)ns/pulse/preanalysis0.h5: $$(Ex1DataNewDir)$(1)/*$(1).root Ex1/$(1)/$(2)ns/h5/preanalysis0.h5 Ex1/$(1)/$(2)ns/h5/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 triggerPulse.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) > $$@.log 2>&1
Ex1/$(1)/$(2)ns/qe/preanalysis%.h5: $$(Ex1DataNewDir)$(1)/*$(1)_%.root Ex1/$(1)/$(2)ns/h5/preanalysis%.h5 Ex1/$(1)/$(2)ns/h5/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 triggerPulse.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) --QE > $$@.log 2>&1
Ex1/$(1)/$(2)ns/qe/preanalysis0.h5: $$(Ex1DataNewDir)$(1)/*$(1).root Ex1/$(1)/$(2)ns/h5/preanalysis0.h5 Ex1/$(1)/$(2)ns/h5/trigger.h5 
	mkdir -p $$(dir $$@)
	time python3 triggerPulse.py -i $$< -c $$(channels) -t $$(triggerch)  -o $$@ --result $$(word 2,$$^) --interval $$(word 3,$$^) --QE > $$@.log 2>&1

Ex1/$(1)/$(2)ns/h5/preanalysisMerge.h5: Ex1/$(1)/$(2)ns/h5/preanalysis0.h5 $$(patsubst %.root,Ex1/$(1)/$(2)ns/h5/preanalysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root))))
	python3 mergeH5.py -i $$^ -o $$@
Ex1/$(1)/$(2)ns/h5/trigger.h5: Ex1/$(1)/$(2)ns/h5/preanalysisMerge.h5
	python summary.py -i $$^ -o $$@ -c $$(channels)
Ex1/$(1)/$(2)ns/pulse/preanalysisMerge.h5: Ex1/$(1)/$(2)ns/pulse/preanalysis0.h5 $$(patsubst %.root,Ex1/$(1)/$(2)ns/pulse/preanalysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root))))
	python3 mergeH5.py -i $$^ -o $$@
Ex1/$(1)/$(2)ns/qe/preanalysisMerge.h5: Ex1/$(1)/$(2)ns/qe/preanalysis0.h5 $$(patsubst %.root,Ex1/$(1)/$(2)ns/qe/preanalysis%.h5,$$(notdir $$(subst $(1)_,/,$$(shell ls -v $$(Ex1DataNewDir)$(1)/*$(1)_*.root))))
	python3 mergeH5.py -i $$^ -o $$@
Ex1/$(1)/$(2)ns/pulse/ratio.pdf: Ex1/$(1)/$(2)ns/pulse/preanalysisMerge.h5
	python3 pulseRatio.py -i $$^ -o $$@ > $$@.log
Ex1/$(1)/$(2)ns/qe/ratio.h5: Ex1/$(1)/$(2)ns/qe/preanalysisMerge.h5 Ex1/$(1)/$(2)ns/h5/trigger.h5
	python3 QERatio.py -i $$< -o $$@ --interval $$(word 2,$$^) -c $$(channels) > $$@.log

Ex1/$(1)/$(2)ns/charge.png: Ex1/$(1)/$(2)ns/h5/preanalysisMerge.h5
	python3 spePrePlot.py -c $$(channels) -t $$(triggerch) -i $$^ -o $$(dir $$@) > $$@.log 2>&1
Ex1/$(1)/$(2)ns/ratio.h5: Ex1/$(1)/$(2)ns/h5/preanalysisMerge.h5
	cp /srv/greatofdream/JinpingPMTTest/pmtTest/trigger/Ex1/580/400ns/config.json $$(dir $$@)
	cp /srv/greatofdream/JinpingPMTTest/pmtTest/trigger/Ex1/580/400ns/config $$(dir $$@)
	python3 QE.py -c $$(channels) -t $$(triggerch) -i $$^ -o $$@ > $$@.log 2>&1
Ex1/$(1)/$(2)ns/0.root: Ex1/$(1)/$(2)ns/h5/preanalysisMerge.h5
	mkdir -p $$(dir $$@)
	python3 h5root.py -c $$(channels) -i $$^ -o $$(dir $$@) > $$@.log 2>&1

endef
$(eval $(foreach an,$(anaNum),$(foreach wc,$(wavecut),$(call analysis,$(an),$(wc)))))
Ex1/401/3500ns/fit2.png: Ex1/401/3500ns/2.root
	root -lbq 'fitChargeRes.C("$^","$@",27,135,120,40, 135)'

Ex1/409/400ns/fit1.png: Ex1/409/400ns/1.root
	root -lbq '../srcuproot3/fitChargeRes.C("$^","$@",4,85,30,20, 50)'
Ex1/410/400ns/fit3.png: Ex1/410/400ns/3.root
	root -lbq '../srcuproot3/fitChargeRes.C("$^","$@",6,170,103,90, 120)'
Ex1/411/400ns/fit3.png: Ex1/411/400ns/3.root
	root -lbq '../srcuproot3/fitChargeRes.C("$^","$@",6,170,120,110, 135)'
Ex1/650/400ns/pulse/ratio.log: Ex1/650/400ns/pulse/ratio.pdf.log
	python3 pulseRatioWOnoise.py -i $< -o $@ --ref ../srcuproot3/Ex1/651/peak.pdf.log
Ex1/654/400ns/pulse/ratio.log: Ex1/654/400ns/pulse/ratio.pdf.log
	python3 pulseRatioWOnoise.py -i $< -o $@ --ref ../srcuproot3/Ex1/653/peak.pdf.log
Ex1/657/400ns/pulse/ratio.log: Ex1/657/400ns/pulse/ratio.pdf.log
	python3 pulseRatioWOnoise.py -i $< -o $@ --ref ../srcuproot3/Ex1/656/peak.pdf.log
.DELETE_ON_ERROR:
.SECONDARY:
