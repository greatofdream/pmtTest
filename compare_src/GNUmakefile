#

include $(SKOFL_ROOT)/config.gmk
LOCAL_LIBS	= -lsklowe_7.0 -lsollib_4.0 -lwtlib_5.1 -lbonsai_3.3

SRCS = $(wildcard ./*.cc)
TARGETS = $(basename $(SRCS))

all: $(TARGETS)

%: %.cc
	$(info ===== Compiling $* =====)
	$(CXX) $(CXXFLAGS) -c $*.cc -o $*.o
	LD_RUN_PATH=$(SKOFL_LIBDIR):$(A_LIBDIR) $(CXX) $(CXXFLAGS) -o $* $*.o $(LDLIBS) 


