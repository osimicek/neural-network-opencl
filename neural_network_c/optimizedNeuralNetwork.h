#ifndef OPTIMIZED_NEURAL_NETWORK_H
#define OPTIMIZED_NEURAL_NETWORK_H
#include "naiveNeuralNetwork.h"
namespace optimized {
    void runOptimizedNeuralNetwork(NeuralNetworkT *neuralNetwork, const char* taskFilename, bool verbose = false);
    void runNeuralNetworkClassification(NeuralNetworkT *neuralNetwork, const char* taskFilename, bool verbose = false, float** classificationResult = NULL);
}
#endif
