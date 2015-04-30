#include <stdio.h>
#include "neural_network_c/optimizedNeuralNetwork.h"
#include "NeuralNetwork.h"
#include "NetworksContainer.h"
#include "NetworksRunner.h"


int compare_networks(NeuralNetworkT *neuralNetwork_c, NeuralNetwork *neuralNetwork_opencl) {
    for (int i = 0; i < neuralNetwork_c->criteria.maxEpochs; i++) {
        // printf("test: %f %f\n", neuralNetwork_c->squareErrorHistory[i], neuralNetwork_opencl->squareErrorHistory[i] * 1.0001);
        if (neuralNetwork_opencl->squareErrorHistory[i] < neuralNetwork_c->squareErrorHistory[i] * 0.9999 ||
            neuralNetwork_opencl->squareErrorHistory[i] > neuralNetwork_c->squareErrorHistory[i] * 1.0001) {
            return 1;
        }
    }
    return 0;
}

int test_11() {
    const char *filename = "./neural_network_c/data/cancer.dt";
    NeuralNetworkT neuralNetwork_c;
    neuralNetwork_c.setup.classification = true;
    neuralNetwork_c.setup.lambda = 1.f;
    neuralNetwork_c.setup.learningFactor = 0.4f;
    neuralNetwork_c.setup.numOfLayers = 3;
    int tmpLayers[] = {-1,  11,  -1};
    neuralNetwork_c.setup.layers = tmpLayers;
    neuralNetwork_c.setup.minOutputValue = 0.f;
    neuralNetwork_c.setup.maxOutputValue = 1.f;
    neuralNetwork_c.criteria.maxEpochs = 10;
    neuralNetwork_c.criteria.minProgress = 5.0f;
    neuralNetwork_c.criteria.maxGeneralizationLoss = 4.0f;

    optimized::runOptimizedNeuralNetwork(&neuralNetwork_c, filename);

    NeuralNetwork neuralNetwork_opencl;
    neuralNetwork_opencl.set_hidden_layers(neuralNetwork_c.setup.numOfLayers - 2, neuralNetwork_c.setup.layers[1]);
    neuralNetwork_opencl.set_learning_factor(neuralNetwork_c.setup.learningFactor);
    neuralNetwork_opencl.set_max_epochs(neuralNetwork_c.criteria.maxEpochs);

    NetworksContainer networks_container;
    std::vector<NeuralNetwork *> * neural_networks = networks_container.get_neural_networks_storage();
    networks_container.set_size(1);
    neural_networks->clear();
    neural_networks->push_back(&neuralNetwork_opencl);

    networks_container.load_input_data(filename);

    NetworksRunner networks_runner(&networks_container);
    networks_runner.write_task_data();
    networks_runner.run_networks();

    return compare_networks(&neuralNetwork_c, &neuralNetwork_opencl);
}

int test_11_11() {
    const char *filename = "./neural_network_c/data/cancer.dt";
    NeuralNetworkT neuralNetwork_c;
    neuralNetwork_c.setup.classification = true;
    neuralNetwork_c.setup.lambda = 1.f;
    neuralNetwork_c.setup.learningFactor = 0.222f;
    neuralNetwork_c.setup.numOfLayers = 4;
    int tmpLayers[] = {-1,  11, 11,  -1};
    neuralNetwork_c.setup.layers = tmpLayers;
    neuralNetwork_c.setup.minOutputValue = 0.f;
    neuralNetwork_c.setup.maxOutputValue = 1.f;
    neuralNetwork_c.criteria.maxEpochs = 10;
    neuralNetwork_c.criteria.minProgress = 5.0f;
    neuralNetwork_c.criteria.maxGeneralizationLoss = 4.0f;

    optimized::runOptimizedNeuralNetwork(&neuralNetwork_c, filename);

    NeuralNetwork neuralNetwork_opencl;
    neuralNetwork_opencl.set_hidden_layers(neuralNetwork_c.setup.numOfLayers - 2, neuralNetwork_c.setup.layers[1]);
    neuralNetwork_opencl.set_learning_factor(neuralNetwork_c.setup.learningFactor);
    neuralNetwork_opencl.set_max_epochs(neuralNetwork_c.criteria.maxEpochs);

    NetworksContainer networks_container;
    std::vector<NeuralNetwork *> * neural_networks = networks_container.get_neural_networks_storage();
    networks_container.set_size(1);
    neural_networks->clear();
    neural_networks->push_back(&neuralNetwork_opencl);

    networks_container.load_input_data(filename);

    NetworksRunner networks_runner(&networks_container);
    networks_runner.write_task_data();
    networks_runner.run_networks();

    return compare_networks(&neuralNetwork_c, &neuralNetwork_opencl);
}



int main(int argc, char **argv)
{   
    int total_tests = 0;
    int failed_tests = 0;
    if (test_11()) {
        printf("11 neurons test failed");
        failed_tests++;
    }
    total_tests++;

    if (test_11_11()) {
        printf("11x11 neurons test failed");
        failed_tests++;
    }
    total_tests++;
    printf("Total tests: %d  failed: %d\n", total_tests, failed_tests);
    return 0;
}