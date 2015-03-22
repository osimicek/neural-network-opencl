ERR = $(shell which icpc >/dev/null; echo $$?)
ifeq "$(ERR)" "0"
    CXX = icpc
    CPPFLAGS += -O3 -xavx
else
    CXX = g++
    CPPFLAGS += -O3
endif



PAPI_EVENTS_F=PAPI_FP_OPS|PAPI_TOT_CYC|PAPI_TOT_INS
PAPI_EVENTS_M=PAPI_L1_DCA|PAPI_L1_DCM|PAPI_L2_DCA|PAPI_L2_DCM
PAPI_EVENTS_M2=PAPI_L1_DCA|PAPI_L1_DCM
PAPI_EVENTS_M_INTEL=PAPI_L1_DCA|PAPI_L1_DCH|PAPI_L1_DCM|PAPI_L1_DCR|PAPI_L1_DCW|PAPI_L2_DCA|PAPI_L2_DCH|PAPI_L2_DCM|PAPI_L2_DCR|PAPI_L2_DCW|PAPI_LSU_IDL

experiment_obj := naiveNeuralNetwork.o \
				  optimizedNeuralNetwork.o
papi_obj := papi_cntr.o\
			initPapi.o

all: join_files $(experiment_obj) main.o
	$(CXX) $(experiment_obj) main.o -o main $(CPPFLAGS) $(OPT)

papi1: papi1_set_flags $(papi_obj) all
papi1_set_flags:
	$(eval experiment_obj = $(papi_obj) $(experiment_obj))
	$(eval CPPFLAGS += -lpapi -L/usr/local/lib64 -DUSE_PAPI_LEARN_AND_TEST=1)

papi2: papi2_set_flags $(papi_obj) all
papi2_set_flags:
	$(eval experiment_obj = $(papi_obj) $(experiment_obj))
	$(eval CPPFLAGS += -lpapi -L/usr/local/lib64 -DUSE_PAPI_LEARN_DETAIL -DUSE_PAPI_TEST_DETAIL)

clean:
	rm -f *.o main

join_files:
	cat data/mnist_part_* > data/mnist.dt

run:
	PAPI_EVENTS='$(PAPI_EVENTS_F)' ./main $(arg)
runm:
	PAPI_EVENTS='$(PAPI_EVENTS_M)' ./main $(arg)
runm2:
	PAPI_EVENTS='$(PAPI_EVENTS_M2)' ./main $(arg)
runintel:
	PAPI_EVENTS='$(PAPI_EVENTS_M_INTEL)' ./main $(arg)
