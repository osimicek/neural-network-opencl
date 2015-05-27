#include <iostream>
#include <stdio.h>
#include <mm_malloc.h>
#include "NeuralNetwork.h"

#define MEMORY_ALIGN 128
#define OPENCL_MEMORY_ALIGN 5 // 2**OPENCL_MEMORY_ALIGN * sizeof(float)


using namespace std;


/**
 * NeuralNetwork data carrier. It doesn't perform any calculations, only initiates and stores
 * settings and state of neuralNetwork.
 */
NeuralNetwork::NeuralNetwork() {
    this->setup.classification = true;
    this->setup.lambda = 1.f;
    this->setup.learningFactor = 0.4f;
    // this->setup.numOfLayers = 4;
    // int tmpLayers[] = {-1, 9, 9, -1};
    this->set_hidden_layers(3, 256);

    this->setup.minOutputValue = 0.f;
    this->setup.maxOutputValue = 1.f;
    this->criteria.maxEpochs = 1;
    this->criteria.minProgress = 5.0f;
    this->criteria.maxGeneralizationLoss = 4.0f;

    this->state.epoch = 0;
    this->state.learningLine = 0;
    this->state.testLine = 0;

    this->allocatedLayers = false;
}

NeuralNetwork::~NeuralNetwork() {
    if (this->allocatedLayers) {
        free(this->setup.layers);
    }
}

/**
 * Prints
 * Structure of neural network 9 3 5 2:
 *
 * values:   ---------
 *           ---
 *           -----
 *           --
 *
 * weights:  --------- | --------- | ---------
 *           --- | --- | --- | --- | ---
 *           ----- | -----
 *
 * expectedOutput: --
 *
 * errors:   #########
 *           ---
 *           -----
 *           --
 *
 * layers:   [9,3,5,2]
 *
 */

void NeuralNetwork::print(float *expectedOutput) {
    int numOfLayers = this->setup.numOfLayers;
    int *layers = this->setup.layers;
    float *values = this->state.values;
    float *weights = this->state.weights;
    float *errors = this->state.errors;

    printf("------------#### Neural network state ####------------\n");
    printf("Values:\n");
    int valueOffset = 0;
    for (int layer = 0; layer < numOfLayers; layer++) {
        int neurons = layers[layer];
        printf("\t");
        for (int neuron = 0; neuron < neurons; neuron++) {
            printf("%6.3f  ", values[neuron + valueOffset]);
        }
        printf("\n");
        valueOffset += neurons;
    }
    printf("Weights:\n");
    int weightOffset = 0;
    for (int layer = 1; layer < numOfLayers; layer++) {
        int neurons = layers[layer];
        int prevNeurons = layers[layer - 1];
        int rounded_layer_size = (((layers[layer] - 1) >> OPENCL_MEMORY_ALIGN) + 1) << OPENCL_MEMORY_ALIGN;
        printf("\t");
        for (int neuron = 0; neuron < neurons; neuron++) {
            for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                printf("%0.2f ",  weights[weightOffset + prevNeuron * rounded_layer_size + neuron]);
                // printf("neur %d ",  weightOffset + prevNeuron);
            }
            printf("| ");
        }
        weightOffset += prevNeurons * rounded_layer_size;
        printf("\n");
    }

    printf("Errors:\n");
    valueOffset = 0;
    for (int layer = 0; layer < numOfLayers; layer++) {
        int neurons = layers[layer];
        printf("\t");
        for (int neuron = 0; neuron < neurons; neuron++) {
            printf("%6.4f  ", errors[neuron + valueOffset]);
        }
        printf("\n");
        valueOffset += neurons;
    }

    printf("Expected output:\n");
    int neurons = layers[numOfLayers - 1];
    printf("\t");
    for (int neuron = 0; neuron < neurons; neuron++) {
        printf("%5.2f   ", expectedOutput[neuron]);
    }
    printf("\n");
    printf("---------------########################---------------\n");
}

/**
 * Imports settings and state from transform structure.
 */
void NeuralNetwork::import_net( neural_network_transform_t *transform,
                                void * neural_network_buffer,
                                void *task_data_buffer,
                                taskData_t *taskData) {
    this->setup.numOfLayers = transform->setup_numOfLayers;
    this->setup.classification = transform->setup_classification;
    this->setup.minOutputValue = transform->setup_minOutputValue;
    this->setup.maxOutputValue = transform->setup_maxOutputValue;
    this->setup.learningFactor = transform->setup_learningFactor;
    this->setup.lambda = transform->setup_lambda;

    this->criteria.maxEpochs = transform->criteria_maxEpochs;
    this->criteria.minProgress = transform->criteria_minProgress;
    this->criteria.maxGeneralizationLoss = transform->criteria_maxGeneralizationLoss;

    this->state.epoch = transform->state_epoch;
    this->state.weights = &((float*)neural_network_buffer)[transform->state_b_offset_weights];
    this->state.values = &((float*)neural_network_buffer)[transform->state_b_offset_values];
    this->state.errors = &((float*)neural_network_buffer)[transform->state_b_offset_errors];
    this->state.testLine = transform->state_testLine;
    this->state.learningLine = transform->state_learningLine;

    this->currentSquareErrorCounter = transform->neuralNetwork_currentSquareErrorCounter;
    this->bestSquareError[0] = transform->neuralNetwork_bestSquareError[0];
    this->bestSquareError[1] = transform->neuralNetwork_bestSquareError[1];
    this->squareErrorHistory = &((float*)neural_network_buffer)[transform->neuralNetwork_b_offset_squareErrorHistory];
}

/**
 * Exports settings and state to transform structure.
 */
void NeuralNetwork::export_net(neural_network_transform_t *transform, void * neural_network_buffer, void *task_data_buffer, taskData_t *taskData) {
    transform->setup_numOfLayers = this->setup.numOfLayers;
    transform->setup_classification = this->setup.classification;
    transform->setup_minOutputValue = this->setup.minOutputValue;
    transform->setup_maxOutputValue = this->setup.maxOutputValue;
    transform->setup_learningFactor = this->setup.learningFactor;
    transform->setup_lambda = this->setup.lambda;

    transform->criteria_maxEpochs = this->criteria.maxEpochs;
    transform->criteria_minProgress = this->criteria.minProgress;
    transform->criteria_maxGeneralizationLoss = this->criteria.maxGeneralizationLoss;

    transform->state_epoch = this->state.epoch;
    transform->state_testLine = this->state.testLine;
    transform->state_learningLine = this->state.learningLine;

    transform->neuralNetwork_currentSquareErrorCounter = this->currentSquareErrorCounter;
    transform->neuralNetwork_bestSquareError[0] = this->bestSquareError[0];
    transform->neuralNetwork_bestSquareError[1] = this->bestSquareError[1];
}

/**
 * Initialization of weights with random values. Seed is fixed to achieve reproducibility.
 */
void NeuralNetwork::init_weights(int length) {
    int *layers = this->setup.layers;
    int weightOffset = 0;
    for (int layer = 1; layer < this->setup.numOfLayers; layer++) {
        int neurons = layers[layer];
        int prevNeurons = layers[layer - 1];
        int rounded_layer_size = (((layers[layer] - 1) >> OPENCL_MEMORY_ALIGN) + 1) << OPENCL_MEMORY_ALIGN;
        for (int neuron = 0; neuron < neurons; neuron++) {
            for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                this->state.weights[weightOffset + prevNeuron * rounded_layer_size + neuron] = (rand() % 100 + 1) / 100.f - 0.5;
                // printf("neur %0.1f ",  this->state.weights[weightOffset + prevNeuron]);
            }
        }
        weightOffset += prevNeurons * rounded_layer_size;
    }
}

/**
 * Returns size of data that must be alocated in GPU. Data like atributes, arrays (values, errors,
 * weights, ...).
 */
int NeuralNetwork::get_required_buffer_size() {
    int numOfWeights = 0;
    int *layers = this->setup.layers;
    for (int i = 1; i < this->setup.numOfLayers; i++) {
        int rounded_layer_size = (((layers[i] - 1) >> OPENCL_MEMORY_ALIGN) + 1) << OPENCL_MEMORY_ALIGN;
        numOfWeights += rounded_layer_size * layers[i - 1];
    }

    int numOfValues = 0;
    for (int i = 0; i < this->setup.numOfLayers; i++) {
        numOfValues += layers[i];
    }
    int numOfSquareErrorHistory = this->criteria.maxEpochs;
    // std::cout << "required buff "<<  (numOfWeights + numOfValues + numOfValues +
    //         numOfSquareErrorHistory + this->setup.numOfLayers) << std::endl;
    int buffer_size = (numOfWeights + numOfValues + numOfValues +
                        numOfSquareErrorHistory + this->setup.numOfLayers);
    return (((buffer_size - 1) >> OPENCL_MEMORY_ALIGN) + 1) << OPENCL_MEMORY_ALIGN;
}

/**
 * Initiates atributes and connect arrays to neural network buffer.
 */
void NeuralNetwork::init(neural_network_transform_t *transform, void *neural_network_buffer) {
    int numOfWeights = 0;
    int *layers = this->setup.layers;
    for (int i = 1; i < this->setup.numOfLayers; i++) {
        int rounded_layer_size = (((layers[i] - 1) >> OPENCL_MEMORY_ALIGN) + 1) << OPENCL_MEMORY_ALIGN;
        numOfWeights += rounded_layer_size * layers[i - 1];
    }
    
    this->bestSquareError[0] = -1.f;
    this->bestSquareError[1] = -1.f;
    this->currentSquareErrorCounter = 0.0f;
    this->state.learningLine = 0;
    this->state.testLine = 0;

    int numOfValues = 0;
    for (int i = 0; i < this->setup.numOfLayers; i++) {
        numOfValues += layers[i];
    }
    int numOfSquareErrorHistory = this->criteria.maxEpochs;


    this->state.weights = (float *) neural_network_buffer;
    this->state.values =  &((float *)neural_network_buffer)[numOfWeights];
    this->state.errors =  &((float *)neural_network_buffer)[numOfValues + numOfWeights];
    this->squareErrorHistory = &((float *)neural_network_buffer)[numOfValues + numOfValues + numOfWeights];
    this->setup.layers = &((int *)neural_network_buffer)[numOfSquareErrorHistory + numOfValues + numOfValues + numOfWeights];
    for (int i = 0; i < this->setup.numOfLayers; i++) {
        this->setup.layers[i] = layers[i];
    }
    if (this->allocatedLayers) {
        free(layers);
        this->allocatedLayers = false;
    }
    transform->state_b_offset_weights = 0;
    transform->state_b_offset_values = numOfWeights;
    transform->state_b_size_values = numOfValues;
    transform->state_b_offset_errors = numOfValues + numOfWeights;
    transform->state_b_size_errors = numOfValues;
    transform->neuralNetwork_b_offset_squareErrorHistory = numOfValues + numOfValues + numOfWeights;
    transform->setup_b_offset_layers = numOfSquareErrorHistory + numOfValues + numOfValues + numOfWeights;
    transform->neuralNetwork_b_size = numOfWeights + numOfValues + numOfValues + numOfSquareErrorHistory + this->setup.numOfLayers;
    transform->setup_numOfLayers = this->setup.numOfLayers;
    this->init_weights(numOfWeights);
}

/**
 * Sets n hidden layers with m neurons each.
 */
void NeuralNetwork::set_hidden_layers(int numberOfHiddenLayers, int numberOfNeuronsPerLayer) {
    this->setup.numOfLayers = 2 + numberOfHiddenLayers;
    this->setup.layers = (int *) _mm_malloc((2 + numberOfHiddenLayers) * sizeof(int), MEMORY_ALIGN);
    this->allocatedLayers = true;
    this->setup.layers[0] = -1;
    this->setup.layers[numberOfHiddenLayers + 1] = -1;

    for (int layer = 1; layer < 1 + numberOfHiddenLayers; layer++) {
        this->setup.layers[layer] = numberOfNeuronsPerLayer;
    }
}

/**
 * Sets maximum of epochs
 */
void NeuralNetwork::set_max_epochs(int epochs) {
    this->criteria.maxEpochs = epochs;
}

/**
 * Sets number of input neurons.
 */
void NeuralNetwork::set_input_layer(int neurons) {
    this->setup.layers[0] = neurons;
}

/**
 * Sets number of output neurons.
 */
void NeuralNetwork::set_output_layer(int neurons) {
    this->setup.layers[this->setup.numOfLayers - 1] = neurons;
}

/**
 * Sets learning factor for neural network.
 */
void NeuralNetwork::set_learning_factor(float learning_factor) {
    this->setup.learningFactor = learning_factor;
}

/**
 * Returns required ammout of shared memory.
 */
int NeuralNetwork::get_required_shared_memory_size() {
    int numOfValues = 0;
    for (int i = 0; i < this->setup.numOfLayers; i++) {
        numOfValues += this->setup.layers[i];
    }
    // number of layers + values + errors + maxEpochs
    // number of values == num of errors
    return this->setup.numOfLayers + numOfValues + numOfValues + this->criteria.maxEpochs;
}

/**
 * Returns best squer error of neural network
 */
float NeuralNetwork::get_best_square_error  () {
    return this->bestSquareError[1];
}

/**
 * Returns best epoch of neural network
 */
int NeuralNetwork::get_best_epoch() {
    return this->bestSquareError[0];
}
  