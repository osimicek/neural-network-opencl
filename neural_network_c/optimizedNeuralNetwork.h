#ifndef OPTIMIZED_NEURAL_NETWORK_H
#define OPTIMIZED_NEURAL_NETWORK_H
#include "naiveNeuralNetwork.h"
namespace optimized {
    void runOptimizedNeuralNetwork(NeuralNetworkT *neuralNetwork, const char* taskFilename, bool verbose = false);
    void runNeuralNetworkPrediction(NeuralNetworkT *neuralNetwork, const char* taskFilename, bool verbose = false);
}
#endif
