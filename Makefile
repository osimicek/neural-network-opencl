EXECUTABLE = main

CC= g++
CPPFLAGS += -Wall -O3 -std=c++0x -lrt -lOpenCL -lglut -lGL -lGLU 

neuralnetwork_obj =    OpenclHelper.o \
						NeuralNetwork.o \
						NetworksContainer.o \
						NetworksRunner.o \
						GeneticAlgorithm.o \

tests_obj =      	$(neuralnetwork_obj)\
					neural_network_c/naiveNeuralNetwork.o\
					neural_network_c/optimizedNeuralNetwork.o\

all: $(EXECUTABLE)

intel: $(neuralnetwork_obj) main.o
	$(CC) $(neuralnetwork_obj) main.o -o  main $(CPPFLAGS)
ifdef OPENCL_INC
  CL_CFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  CL_LDFLAGS = -L$(OPENCL_LIB)
endif

main: $(neuralnetwork_obj) $(EXECUTABLE).o
	$(CC) $(neuralnetwork_obj) $(EXECUTABLE).o -o  $(EXECUTABLE) $(CPPFLAGS)

tests: .tests

info:
	./main -i

bench:
	. ./bench_tests.sh | tee bench_output.txt

.tests: $(tests_obj) tests.o
	$(CC) $(tests_obj) tests.o -o  tests $(CPPFLAGS)
	./tests

run:
	./main

try: clean all run

clean:
	rm -f $(EXECUTABLE) tests *.o neural_network_c/*.o
