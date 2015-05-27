EXECUTABLE = main
DEFAULT_DEVICE_ID = 0
DEFAULT_PLATFORM_ID = 0

CC= g++
CPPFLAGS += -w -O3 -std=c++0x -lrt -lOpenCL -lglut -lGL -lGLU -DDEVICE_ID=$(DEFAULT_DEVICE_ID) -DPLATFORM_ID=$(DEFAULT_PLATFORM_ID)

neuralnetwork_obj =    OpenclHelper.o \
						NeuralNetwork.o \
						NetworkContainer.o \
						NetworkRunner.o \
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

info:
	./main -i

bench:
	. ./bench_tests.sh | tee bench_output.txt

tests: .tests
	./tests

.tests: $(tests_obj) tests.o
	$(CC) $(tests_obj) tests.o -o  tests $(CPPFLAGS)

example1:
	./main -b -w 128 -n 128 -e 5 -t ./data/cancer.dt -c ./data/cancer_classification.dt -o out.txt

example2:
	./main -b -w 256 -n 256 -l 5 -e 3 -t ./data/cancer.dt -c ./data/cancer_classification.dt -o out.txt

example3:
	./main -g 5 -n 128 -e 2 -t ./data/gene.dt -c ./data/gene_classification.dt -o out.txt

example4:
	./main -g 5 -n 128 -x 1 -e 2 -t ./data/gene.dt -c ./data/gene_classification.dt -o out.txt

run:
	./main

try: clean all run

clean:
	rm -f $(EXECUTABLE) tests *.o neural_network_c/*.o
