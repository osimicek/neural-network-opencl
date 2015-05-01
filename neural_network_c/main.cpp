#include <stdio.h>
#include "initPapi.h"
#include "naiveNeuralNetwork.h"
#include <unistd.h>
#include "optimizedNeuralNetwork.h"



void test(int argc, char **argv) {
    char *filename;
    if (argc < 2) {
        const char *default_filename = "./data/cancer.dt";
        filename = (char *) default_filename;
    } else {
        filename = argv[1];
    }

    NeuralNetworkT neuralNetwork;

    neuralNetwork.setup.classification = true;
    neuralNetwork.setup.lambda = 1.f;
    neuralNetwork.setup.learningFactor = 0.4f;
    // neuralNetwork.setup.numOfLayers = 4;
    // int tmpLayers[] = {-1, 9, 9, -1};
    neuralNetwork.setup.numOfLayers = 5;
    int tmpLayers[] = {-1,  32,32,32,  -1};
    neuralNetwork.setup.layers = tmpLayers;

    neuralNetwork.setup.minOutputValue = 0.f;
    neuralNetwork.setup.maxOutputValue = 1.f;
    // neuralNetwork.criteria.maxEpochs = 250;
    neuralNetwork.criteria.maxEpochs = 1000;
    neuralNetwork.criteria.minProgress = 5.0f;
    neuralNetwork.criteria.maxGeneralizationLoss = 4.0f;
    // for (float l = 0.1; l < 1; l +=0.1) {
    //     naive::runNaiveNeuralNetwork(argv[1], l);
    // }
    optimized::runOptimizedNeuralNetwork(&neuralNetwork, filename);
}


int main(int argc, char **argv)
{   
    int number_of_processes = 1;
    char *filename;
    if (argc < 2) {
        const char *default_filename = "./data/cancer.dt";
        filename = (char *) default_filename;
    } else {
        filename = argv[1];
    }

    #if USE_PAPI_LEARN_AND_TEST || USE_PAPI_LEARN_DETAIL || USE_PAPI_TEST_DETAIL || USE_PAPI_NEURAL_ROW_LEARN || USE_PAPI_NEURAL_ROW_TEST
    initPapi();
    #endif


    for (int i = 1; i < number_of_processes; i++) {
        if(fork() != 0) break;
    }

    test(argc, argv);
    // naive::runNaiveNeuralNetwork(&neuralNetwork, filename);
    #if USE_PAPI_LEARN_AND_TEST || USE_PAPI_LEARN_DETAIL || USE_PAPI_TEST_DETAIL || USE_PAPI_NEURAL_ROW_LEARN || USE_PAPI_NEURAL_ROW_TEST
    papi_routines->PrintScreen();
    delete papi_routines;
    #endif
    return 0;
}
