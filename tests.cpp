#include <stdio.h>
#include "neural_network_c/optimizedNeuralNetwork.h"
#include "NeuralNetwork.h"
#include "NetworkContainer.h"
#include "NetworkRunner.h"

#ifndef DEVICE_ID
    #define DEVICE_ID 0
    #define PLATFORM_ID 0
#endif

int compare_networks(NeuralNetworkT *neuralNetwork_c, NeuralNetwork *neuralNetwork_opencl) {
    for (int i = 0; i < neuralNetwork_c->criteria.maxEpochs; i++) {
        // printf("test: %f %f\n", neuralNetwork_c->squareErrorHistory[i], neuralNetwork_opencl->squareErrorHistory[i] );
        if (neuralNetwork_opencl->squareErrorHistory[i] < neuralNetwork_c->squareErrorHistory[i] * 0.9999 ||
            neuralNetwork_opencl->squareErrorHistory[i] > neuralNetwork_c->squareErrorHistory[i] * 1.0001) {
            return 1;
        }
    }
    return 0;
}

int test_11() {
    const char *filename = "./data/cancer.dt";
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

    srand(37);
    optimized::runOptimizedNeuralNetwork(&neuralNetwork_c, filename);

    srand(37);
    NeuralNetwork neuralNetwork_opencl;
    neuralNetwork_opencl.set_hidden_layers(neuralNetwork_c.setup.numOfLayers - 2, neuralNetwork_c.setup.layers[1]);
    neuralNetwork_opencl.set_learning_factor(neuralNetwork_c.setup.learningFactor);
    neuralNetwork_opencl.set_max_epochs(neuralNetwork_c.criteria.maxEpochs);

    NetworkContainer networks_container;
    std::vector<NeuralNetwork *> * neural_networks = networks_container.get_neural_networks_storage();
    networks_container.set_size(1);
    neural_networks->clear();
    neural_networks->push_back(&neuralNetwork_opencl);

    networks_container.load_input_data(filename);

    NetworkRunner networks_runner(PLATFORM_ID, DEVICE_ID, &networks_container);
    networks_runner.write_task_data();
    networks_runner.run_networks();

    return compare_networks(&neuralNetwork_c, &neuralNetwork_opencl);
}

int test_80_80() {
    const char *filename = "./data/cancer.dt";
    NeuralNetworkT neuralNetwork_c;
    neuralNetwork_c.setup.classification = true;
    neuralNetwork_c.setup.lambda = 1.f;
    neuralNetwork_c.setup.learningFactor = 0.222f;
    neuralNetwork_c.setup.numOfLayers = 4;
    int tmpLayers[] = {-1,  80, 80,  -1};
    neuralNetwork_c.setup.layers = tmpLayers;
    neuralNetwork_c.setup.minOutputValue = 0.f;
    neuralNetwork_c.setup.maxOutputValue = 1.f;
    neuralNetwork_c.criteria.maxEpochs = 10;
    neuralNetwork_c.criteria.minProgress = 5.0f;
    neuralNetwork_c.criteria.maxGeneralizationLoss = 4.0f;

    srand(37);
    optimized::runOptimizedNeuralNetwork(&neuralNetwork_c, filename);

    srand(37);
    NeuralNetwork neuralNetwork_opencl;
    neuralNetwork_opencl.set_hidden_layers(neuralNetwork_c.setup.numOfLayers - 2, neuralNetwork_c.setup.layers[1]);
    neuralNetwork_opencl.set_learning_factor(neuralNetwork_c.setup.learningFactor);
    neuralNetwork_opencl.set_max_epochs(neuralNetwork_c.criteria.maxEpochs);

    NetworkContainer networks_container;
    std::vector<NeuralNetwork *> * neural_networks = networks_container.get_neural_networks_storage();
    networks_container.set_size(1);
    neural_networks->clear();
    neural_networks->push_back(&neuralNetwork_opencl);

    networks_container.load_input_data(filename);

    NetworkRunner networks_runner(PLATFORM_ID, DEVICE_ID, &networks_container);
    networks_runner.write_task_data();
    networks_runner.run_networks();

    return compare_networks(&neuralNetwork_c, &neuralNetwork_opencl);
}

int test_11_8() {
    const char *filename = "./data/cancer.dt";
    NeuralNetworkT neuralNetwork_c;
    neuralNetwork_c.setup.classification = true;
    neuralNetwork_c.setup.lambda = 1.f;
    neuralNetwork_c.setup.learningFactor = 0.222f;
    neuralNetwork_c.setup.numOfLayers = 4;
    int tmpLayers[] = {-1,  11, 8,  -1};
    neuralNetwork_c.setup.layers = tmpLayers;
    neuralNetwork_c.setup.minOutputValue = 0.f;
    neuralNetwork_c.setup.maxOutputValue = 1.f;
    neuralNetwork_c.criteria.maxEpochs = 10;
    neuralNetwork_c.criteria.minProgress = 5.0f;
    neuralNetwork_c.criteria.maxGeneralizationLoss = 4.0f;

    srand(37);
    optimized::runOptimizedNeuralNetwork(&neuralNetwork_c, filename);

    srand(37);
    NeuralNetwork neuralNetwork_opencl;
    neuralNetwork_opencl.set_hidden_layers(neuralNetwork_c.setup.numOfLayers - 2, neuralNetwork_c.setup.layers[1]);
    neuralNetwork_opencl.setup.layers[2] = 8;
    neuralNetwork_opencl.set_learning_factor(neuralNetwork_c.setup.learningFactor);
    neuralNetwork_opencl.set_max_epochs(neuralNetwork_c.criteria.maxEpochs);

    NetworkContainer networks_container;
    std::vector<NeuralNetwork *> * neural_networks = networks_container.get_neural_networks_storage();
    networks_container.set_size(1);
    neural_networks->clear();
    neural_networks->push_back(&neuralNetwork_opencl);

    networks_container.load_input_data(filename);

    NetworkRunner networks_runner(PLATFORM_ID, DEVICE_ID, &networks_container);
    networks_runner.write_task_data();
    networks_runner.run_networks();

    return compare_networks(&neuralNetwork_c, &neuralNetwork_opencl);
}


int test_container() {
    const char *filename = "./data/cancer.dt";
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
    srand(37);
    optimized::runOptimizedNeuralNetwork(&neuralNetwork_c, filename);
    NeuralNetworkT neuralNetwork_c_2;
    neuralNetwork_c_2.setup.classification = true;
    neuralNetwork_c_2.setup.lambda = 1.f;
    neuralNetwork_c_2.setup.learningFactor = 0.4f;
    neuralNetwork_c_2.setup.numOfLayers = 3;
    int tmpLayers_2[] = {-1,  22,  -1};
    neuralNetwork_c_2.setup.layers = tmpLayers_2;
    neuralNetwork_c_2.setup.minOutputValue = 0.f;
    neuralNetwork_c_2.setup.maxOutputValue = 1.f;
    neuralNetwork_c_2.criteria.maxEpochs = 10;
    neuralNetwork_c_2.criteria.minProgress = 5.0f;
    neuralNetwork_c_2.criteria.maxGeneralizationLoss = 4.0f;
    srand(37);
    optimized::runOptimizedNeuralNetwork(&neuralNetwork_c_2, filename);

    NetworkContainer networks_container;
    std::vector<NeuralNetwork *> * neural_networks = networks_container.get_neural_networks_storage();
    networks_container.set_size(4);
    neural_networks->clear();
    srand(37);
    NeuralNetwork neuralNetwork_opencl_1;
    neuralNetwork_opencl_1.set_hidden_layers(neuralNetwork_c.setup.numOfLayers - 2, neuralNetwork_c.setup.layers[1]);
    neuralNetwork_opencl_1.set_learning_factor(neuralNetwork_c.setup.learningFactor);
    neuralNetwork_opencl_1.set_max_epochs(neuralNetwork_c.criteria.maxEpochs);
    neural_networks->push_back(&neuralNetwork_opencl_1);

    srand(37);
    NeuralNetwork neuralNetwork_opencl_2;
    neuralNetwork_opencl_2.set_hidden_layers(neuralNetwork_c.setup.numOfLayers - 2, neuralNetwork_c.setup.layers[1]);
    neuralNetwork_opencl_2.set_learning_factor(neuralNetwork_c.setup.learningFactor);
    neuralNetwork_opencl_2.set_max_epochs(neuralNetwork_c.criteria.maxEpochs);
    neural_networks->push_back(&neuralNetwork_opencl_2);

    srand(37);
    NeuralNetwork neuralNetwork_opencl_3;
    neuralNetwork_opencl_3.set_hidden_layers(neuralNetwork_c.setup.numOfLayers - 2, neuralNetwork_c.setup.layers[1]);
    neuralNetwork_opencl_3.set_learning_factor(neuralNetwork_c.setup.learningFactor);
    neuralNetwork_opencl_3.set_max_epochs(neuralNetwork_c.criteria.maxEpochs);
    neural_networks->push_back(&neuralNetwork_opencl_3);

    srand(37);
    NeuralNetwork neuralNetwork_opencl_4;
    neuralNetwork_opencl_4.set_hidden_layers(neuralNetwork_c_2.setup.numOfLayers - 2, neuralNetwork_c_2.setup.layers[1]);
    neuralNetwork_opencl_4.set_learning_factor(neuralNetwork_c_2.setup.learningFactor);
    neuralNetwork_opencl_4.set_max_epochs(neuralNetwork_c_2.criteria.maxEpochs);
    neural_networks->push_back(&neuralNetwork_opencl_4);

    networks_container.load_input_data(filename);

    NetworkRunner networks_runner(PLATFORM_ID, DEVICE_ID, &networks_container);
    networks_runner.write_task_data();
    networks_runner.run_networks();

    return compare_networks(&neuralNetwork_c, &neuralNetwork_opencl_1) && compare_networks(&neuralNetwork_c_2, &neuralNetwork_opencl_4);
}


int test_classification() {
    const char *filename = "./data/cancer.dt";
    const char *classification_task_filename = "./data/cancer_classification.dt";
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

    srand(37);
    optimized::runOptimizedNeuralNetwork(&neuralNetwork_c, filename);
    float *class_result;
    optimized::runNeuralNetworkClassification(&neuralNetwork_c, classification_task_filename, false, &class_result);

    srand(37);
    NeuralNetwork neuralNetwork_opencl;
    neuralNetwork_opencl.set_hidden_layers(neuralNetwork_c.setup.numOfLayers - 2, neuralNetwork_c.setup.layers[1]);
    neuralNetwork_opencl.set_learning_factor(neuralNetwork_c.setup.learningFactor);
    neuralNetwork_opencl.set_max_epochs(neuralNetwork_c.criteria.maxEpochs);

    NetworkContainer networks_container;
    std::vector<NeuralNetwork *> * neural_networks = networks_container.get_neural_networks_storage();
    networks_container.set_size(1);
    neural_networks->clear();
    neural_networks->push_back(&neuralNetwork_opencl);

    networks_container.load_input_data(filename);

    NetworkRunner networks_runner(PLATFORM_ID, DEVICE_ID, &networks_container);
    networks_runner.write_task_data();
    networks_runner.run_networks();

    networks_container.load_classification_data(classification_task_filename);
    networks_runner.write_task_data();
    networks_runner.run_networks_classification(64);

    for (uint row = 0; row < networks_container.taskData.totalLearningLines; row++) {
        int numOfOutputNeurons = networks_container.outputVectorSize;
        for (uint output = 0; output < numOfOutputNeurons; output++) {
            int pos = output + row * numOfOutputNeurons;
            // printf("%f %f\n", networks_container.taskData.learningOutputs[pos], class_result[pos]);
            if (networks_container.taskData.learningOutputs[pos] != class_result[pos]) {

                return 1;
            }
        }
    }
    return 0;
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

    if (test_80_80()) {
        printf("80x80 neurons test failed");
        failed_tests++;
    }
    total_tests++;

    if (test_11_8()) {
        printf("11x8 neurons test failed");
        failed_tests++;
    }
    total_tests++;

    if (test_container()) {
        printf("Container test failed");
        failed_tests++;
    }
    total_tests++;

    if (test_classification()) {
        printf("Classification test failed");
        failed_tests++;
    }
    total_tests++;
    printf("Total tests: %d  failed: %d\n", total_tests, failed_tests);
    return 0;
}