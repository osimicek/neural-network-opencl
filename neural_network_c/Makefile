ERR = $(shell which icpc >/dev/null; echo $$?)
ifeq "$(ERR)" "0"
    CXX = icpc
    CPPFLAGS += -O3 -msse4.2 -std=c++0x -qopt-report3
else
    CXX = g++
    CPPFLAGS += -O3 -std=c++0x
endif



PAPI_EVENTS_F=PAPI_FP_OPS|PAPI_TOT_INS|PAPI_TOT_CYC
PAPI_EVENTS_M=PAPI_L1_DCA|PAPI_L1_DCM|PAPI_L2_DCA|PAPI_L2_DCM
PAPI_EVENTS_M2=PAPI_L1_DCA|PAPI_L1_DCM
PAPI_EVENTS_F_INTEL=PAPI_FP_OPS|PAPI_FP_INS|PAPI_SP_OPS|PAPI_VEC_SP|PAPI_TOT_INS
PAPI_EVENTS_M_INTEL=PAPI_LD_INS|PAPI_SR_INS|PAPI_TOT_CYC

experiment_obj := naiveNeuralNetwork.o \
				  optimizedNeuralNetwork.o
papi_obj := papi_cntr.o\
			initPapi.o

all: $(experiment_obj) main.o
	$(CXX) $(experiment_obj) main.o -o main $(CPPFLAGS) $(OPT)

papi_overall: papi_overall_set_flags $(papi_obj) all
papi_overall_set_flags:
	$(eval experiment_obj = $(papi_obj) $(experiment_obj))
	$(eval CPPFLAGS += -lpapi -L/usr/local/lib64 -DUSE_PAPI_LEARN_AND_TEST=1)

papi_detail: papi_detail_set_flags $(papi_obj) all
papi_detail_set_flags:
	$(eval experiment_obj = $(papi_obj) $(experiment_obj))
	$(eval CPPFLAGS += -lpapi -L/usr/local/lib64 -DUSE_PAPI_LEARN_DETAIL=1 -DUSE_PAPI_TEST_DETAIL=1)

papi_neural_row: papi_neural_row_set_flags $(papi_obj) all
papi_neural_row_set_flags:
	$(eval experiment_obj = $(papi_obj) $(experiment_obj))
	$(eval CPPFLAGS += -lpapi -L/usr/local/lib64 -DUSE_PAPI_NEURAL_ROW_LEARN=1 -DUSE_PAPI_NEURAL_ROW_TEST=1)

clean:
	rm -f *.o *.optrpt main

run:
	PAPI_EVENTS='$(PAPI_EVENTS_F)' ./main $(arg)
runm:
	PAPI_EVENTS='$(PAPI_EVENTS_M)' ./main $(arg)
runm2:
	PAPI_EVENTS='$(PAPI_EVENTS_M2)' ./main $(arg)
runintel:
	PAPI_EVENTS='$(PAPI_EVENTS_F_INTEL)' ./main $(arg)
runintelm:
	PAPI_EVENTS='$(PAPI_EVENTS_M_INTEL)' ./main $(arg)
