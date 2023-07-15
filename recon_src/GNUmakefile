#

include $(SKOFL_ROOT)/config.gmk
FCFLAGS:=$(if $(tentative),$(FCFLAGS) -DTENTATIVE,$(FCFLAGS))
tentativeFlag:=$(if $(tentative),_tentative,)
LOCAL_LIBS	= lfneweff_sk6.o lecompte.o -lsklowe_7.0 -lsollib_4.0 -lwtlib_5.1 -lbonsai_3.3 -lstmu -lska
# LOCAL_LIBS	= trabs_sk5.o lfneweff_sk5.o lecompte.o -lsklowe_7.0 -lsollib_4.0 -lwtlib_5.1 -lbonsai_3.3 -lstmu -lska

OBJS   = lfneweff_sk6.o lecompte.o lowfit_mc lowfit_data
# OBJS   = trabs_sk5.o lfneweff_sk5.o lecompte.o lowfit_mc lowfit_data

all: $(OBJS) lowfit_mc$(tentativeFlag) lowfit_data$(tentativeFlag)
 
lowfit_mc_tentative: lowfit_mc
	mv $^ $@
lowfit_data_tentative: lowfit_data
	mv $^ $@
clean: 
	$(RM) *.o *~ core fort.* *.log $(OBJS) output.dat*
