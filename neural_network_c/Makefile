CC= g++
CPPFLAGS += -lpapi -L/usr/local/lib64 -O3
REPORT=

PAPI_EVENTS_F=PAPI_FP_OPS|PAPI_TOT_CYC|PAPI_TOT_INS
PAPI_EVENTS_M=PAPI_L1_DCA|PAPI_L1_DCM|PAPI_L2_DCA|PAPI_L2_DCM
PAPI_EVENTS_M2=PAPI_L1_DCA|PAPI_L1_DCM
PAPI_EVENTS_M_INTEL=PAPI_L1_DCA|PAPI_L1_DCH|PAPI_L1_DCM|PAPI_L1_DCR|PAPI_L1_DCW|PAPI_L2_DCA|PAPI_L2_DCH|PAPI_L2_DCM|PAPI_L2_DCR|PAPI_L2_DCW|PAPI_LSU_IDL

experiment_obj := papi_cntr.o \
				  initPapi.o \
				  naiveNeuralNetwork.o \

all: $(experiment_obj) main.o
	$(CC) $(experiment_obj) main.o -o main $(CPPFLAGS) $(OPT)

papi1: papi1setflags all
papi1setflags:
	$(eval CPPFLAGS += -DUSE_PAPI_LEARN_AND_TEST=1)

papi2: papi2setflags all
papi2setflags:
	$(eval CPPFLAGS +=  -DUSE_PAPI_LEARN_DETAIL -DUSE_PAPI_TEST_DETAIL)

clean:
	rm -f *.o main

run:
	PAPI_EVENTS='$(PAPI_EVENTS_F)' ./main
runm:
	PAPI_EVENTS='$(PAPI_EVENTS_M)' ./main
runm2:
	PAPI_EVENTS='$(PAPI_EVENTS_M2)' ./main
runintel:
	PAPI_EVENTS='$(PAPI_EVENTS_M_INTEL)' ./main
