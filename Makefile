EXECUTABLES = main

CC= g++
CPPFLAGS += -Wall -std=c++0x -lrt -lOpenCL -lglut -lGL -lGLU 

neuralnetwork_obj :=    OpenclHelper.o \
						NeuralNetwork.o \
						NetworksContainer.o \
						NetworksRunner.o \
						GeneticAlgorithm.o \

all: $(EXECUTABLES)

intel: $(neuralnetwork_obj) main.o
	$(CC) $(neuralnetwork_obj) main.o -o  main $(CPPFLAGS)
ifdef OPENCL_INC
  CL_CFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  CL_LDFLAGS = -L$(OPENCL_LIB)
endif

main: $(neuralnetwork_obj) $(EXECUTABLES).o
	$(CC) $(neuralnetwork_obj) $(EXECUTABLES).o -o  $(EXECUTABLES) $(CPPFLAGS)

run:
	./main

try: clean all run

clean:
	rm -f $(EXECUTABLES) *.o
