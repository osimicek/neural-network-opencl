#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include "initPapi.h"
#include "naiveNeuralNetwork.h"
#include "optimizedNeuralNetwork.h"
using namespace naive;

namespace optimized {
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

    void printNeuralNetwork(NeuralNetwork *neuralNetwork, float *expectedOutput) {
        int numOfLayers = neuralNetwork->setup.numOfLayers;
        int *layers = neuralNetwork->setup.layers;
        float *values = neuralNetwork->state.values;
        float *weights = neuralNetwork->state.weights;
        float *errors = neuralNetwork->state.errors;

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
            printf("\t");
            for (int neuron = 0; neuron < neurons; neuron++) {
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    printf("%0.1f ",  weights[weightOffset + prevNeuron]);
                }
                printf("| ");
                weightOffset += prevNeurons;
            }
            printf("\n");
        }

        printf("Errors:\n");
        valueOffset = 0;
        for (int layer = 0; layer < numOfLayers; layer++) {
            int neurons = layers[layer];
            printf("\t");
            for (int neuron = 0; neuron < neurons; neuron++) {
                printf("%6.3f  ", errors[neuron + valueOffset]);
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
     * Initialization of weights with random values. Seed is fixed to achieve reproducibility.
     */
    void initWeights(float *weights, int length) {
        srand(37);
        for (int i = 0; i < length; i++) {
            weights[i] = rand() % 4 + 1;
        }
    }



    /**
     * Returns sum of squareErrors
     */
    float squareErrorSum(NeuralNetwork *neuralNetwork) {
        float sum = 0.0f;
        for (int i = 0; i <= neuralNetwork->state.epoch; i++) {
            sum += neuralNetwork->squareErrorHistory[i];
        }
        return sum;
    }




    /**
     * Initialization of neural network
     */
    void initNeuralNetwork(NeuralNetwork *neuralNetwork) {
        int numOfWeights = 0;
        for (int i = 1; i < neuralNetwork->setup.numOfLayers; i++) {
            numOfWeights += neuralNetwork->setup.layers[i] * neuralNetwork->setup.layers[i - 1];
        }

        neuralNetwork->state.weights = (float *) malloc (numOfWeights * sizeof(float));
        initWeights(neuralNetwork->state.weights, numOfWeights);

        int classificationAccurancyLength = 4 * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * neuralNetwork->criteria.maxEpochs;
        if (neuralNetwork->setup.classification) {
            neuralNetwork->classificationAccurancyHistory = (int *) malloc (classificationAccurancyLength * sizeof(int));
        }
        initAccurancy(neuralNetwork->classificationAccurancyHistory, classificationAccurancyLength);

        neuralNetwork->squareErrorHistory = (float *) malloc (2 * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * neuralNetwork->criteria.maxEpochs * sizeof(int));
        neuralNetwork->bestSquareError[0] = -1.f;
        neuralNetwork->bestSquareError[1] = -1.f;
        neuralNetwork->currentSquareErrorCounter = 0.0f;

        int numOfValues = 0;
        for (int i = 0; i < neuralNetwork->setup.numOfLayers; i++) {
            numOfValues += neuralNetwork->setup.layers[i];
        }
        neuralNetwork->state.values =  (float *) malloc (numOfValues * sizeof(float));
        neuralNetwork->state.errors =  (float *) malloc (numOfValues * sizeof(float));
    }


    /**
     * Performs a learning cycle of neural network. Input vector is transformed to output vector.
     * Output vector si compared with expected output and error is backpropagated throw net and
     * weights are modified.
     */
    void neuralLearnCycle(NeuralNetwork *neuralNetwork, 
                    float *expectedOutput) {
        int numOfLayers = neuralNetwork->setup.numOfLayers;
        int *layers = neuralNetwork->setup.layers;
        float *values = neuralNetwork->state.values;
        float *weights = neuralNetwork->state.weights;
        float *errors = neuralNetwork->state.errors;


        int valueOffset = 0;
        int prevValueOffset = 0;
        int weightOffset = 0;

        // foward value computation
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_learning_foward_computation"].Start();
        #endif
        for (int layer = 1; layer < numOfLayers; layer++) {
            int neurons = layers[layer];
            int prevNeurons = layers[layer - 1];
            valueOffset += prevNeurons;
            for (int neuron = 0; neuron < neurons; neuron++) {
                float value = 0;
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    value += values[prevNeuron + prevValueOffset] * weights[prevNeuron + weightOffset];
                }
                // std::cout << value << "  ";
                values[neuron + valueOffset] = 1. / (1. + exp(-neuralNetwork->setup.lambda * value));
                weightOffset += prevNeurons;
            }
            prevValueOffset += prevNeurons;
        }
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_learning_foward_computation"].Stop();
        #endif

        // backwards error computation
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_learning_error_computation"].Start();
        #endif
        for (int neuron = 0; neuron < layers[numOfLayers - 1]; neuron++) {
            float value = values[neuron + valueOffset];

            float cmpValue = value;
            if (neuralNetwork->setup.classification) {
                cmpValue = round(value);
            }

            if (expectedOutput[neuron] - cmpValue) {
                // std::cout << "chyba u " << neuron << " " << value << std::endl;
                float ex = 1.f / (1.f + exp(-neuralNetwork->setup.lambda * value));;
                // std::cout << "velikost ch " << (expectedOutput[neuron] - value) * (ex * (1 - ex)) << " ex " << ex << std::endl;
                errors[neuron + valueOffset] = (expectedOutput[neuron] - value) * (ex * (1 - ex));
            } else {
                errors[neuron + valueOffset] = 0.f;
            }
            // std::cout << (expectedOutput[neuron] - value) << std::endl;
        }
        // std::cout  << std::endl;
        int followValueOffset = valueOffset;
        for (int layer = numOfLayers - 2; layer > 0; layer--) {
            int neurons = layers[layer];
            int followNeurons = layers[layer + 1];
            valueOffset -= neurons;
            weightOffset -= neurons * followNeurons;
            for (int neuron = 0; neuron < neurons; neuron++) {
                float weightError = 0.f;
                for (int followNeuron = 0; followNeuron < followNeurons; followNeuron++) {
                    weightError += errors[followNeuron + followValueOffset] *
                                  weights[followNeuron + weightOffset];
                }
                weightOffset += followNeurons;
                float value = values[neuron + valueOffset];
                float ex = exp(-neuralNetwork->setup.lambda*value);
                errors[neuron + valueOffset] = weightError * (ex * (1 - ex));
            }
            weightOffset -= neurons * followNeurons;
            followValueOffset -= neurons;
        }
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_learning_error_computation"].Stop();
        #endif
        // printNeuralNetwork(neuralNetwork, expectedOutput);
        // error propagation
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_learning_weight_update"].Start();
        #endif
        weightOffset = 0;
        valueOffset = 0;
        for (int layer = 1; layer < numOfLayers; layer++) {
            int neurons = layers[layer];
            int prevNeurons = layers[layer - 1];
            for (int neuron = 0; neuron < neurons; neuron++) {
                float value = 0.f;
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    weights[prevNeuron + weightOffset] = weights[prevNeuron + weightOffset] +
                                                         neuralNetwork->setup.learningFactor * values[prevNeuron + valueOffset] *
                                                         errors[neuron + valueOffset + prevNeurons];
                }
                weightOffset += prevNeurons;
            }
            valueOffset += prevNeurons;
        }
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_learning_weight_update"].Stop();
        #endif

        // printNeuralNetwork(neuralNetwork, expectedOutput);

        
    }

    /**
     * Performs a test cycle of neural network. Input vector is transformed to output vector.
     * Output vector si compared with expected output. Accurancy is counted from this error.
     */
    void neuralTestCycle(NeuralNetwork *neuralNetwork, 
                    float *expectedOutput) {
        
        int numOfLayers = neuralNetwork->setup.numOfLayers;
        int *layers = neuralNetwork->setup.layers;
        float *values = neuralNetwork->state.values;
        float *weights = neuralNetwork->state.weights;

        int valueOffset = 0;
        int prevValueOffset = 0;
        int weightOffset = 0;

        // foward value computation
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_testing_foward_computation"].Start();
        #endif
        for (int layer = 1; layer < numOfLayers; layer++) {
            int neurons = layers[layer];
            int prevNeurons = layers[layer - 1];
            valueOffset += prevNeurons;
            for (int neuron = 0; neuron < neurons; neuron++) {
                float value = 0.f;
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    value += values[prevNeuron + prevValueOffset] * weights[prevNeuron + weightOffset];
                }
                // std::cout << value << "  ";
                values[neuron + valueOffset] = 1.f / (1.f + exp(-neuralNetwork->setup.lambda * value));
                weightOffset += prevNeurons;
            }
            prevValueOffset += prevNeurons;
        }

        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_testing_foward_computation"].Stop();
        #endif


        // error computation backwards
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_testing_error_computation"].Start();
        #endif
        int numOfOutputNeurons = layers[numOfLayers - 1];
        for (int neuron = 0; neuron < numOfOutputNeurons; neuron++) {
            float value = values[neuron + valueOffset];
            float diff;
            

            if (neuralNetwork->setup.classification) {
                // conversion to bool
                int boolValue = round(value);
                diff = expectedOutput[neuron] - value;
                // neuralNetwork->currentSquareErrorCounter += diff * diff; 

                int pos = 4 * neuron + 4 * numOfOutputNeurons * neuralNetwork->state.epoch;
                if (boolValue == 1 && expectedOutput[neuron] == 1) {
                    // true positive
                    neuralNetwork->classificationAccurancyHistory[pos] += 1;
                } else if (boolValue == 1 && expectedOutput[neuron] == 0) {
                    // true negative
                    // diff = expectedOutput[neuron] - value;
                    neuralNetwork->currentSquareErrorCounter += 1; 

                    neuralNetwork->classificationAccurancyHistory[pos + 1] += 1;
                } else if (boolValue == 0 && expectedOutput[neuron] == 1) {
                    // false negative
                    // diff = expectedOutput[neuron] - value;
                    neuralNetwork->currentSquareErrorCounter += 1; 

                    neuralNetwork->classificationAccurancyHistory[pos + 2] += 1;
                } else if (boolValue == 0 && expectedOutput[neuron] == 0) {
                    // false positive
                    neuralNetwork->classificationAccurancyHistory[pos + 3] += 1; 
                }
            } else {
                diff = expectedOutput[neuron] - value;
                neuralNetwork->currentSquareErrorCounter += diff * diff; 
            }
        }
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_testing_error_computation"].Stop();
        #endif 
    }

    /**
     * Performs trainign and testing of neural network.
     */
    void runOptimizedNeuralNetwork(float learningFactor) {
        NeuralNetwork neuralNetwork;
        TaskData taskData;

        neuralNetwork.setup.classification = true;
        neuralNetwork.setup.lambda = 1.f;
        neuralNetwork.setup.learningFactor = learningFactor;
        neuralNetwork.setup.numOfLayers = 4;
        int tmpLayers[] = {9, 9, 9, 2};
        neuralNetwork.setup.layers = tmpLayers;

        neuralNetwork.setup.minOutputValue = 0.f;
        neuralNetwork.setup.maxOutputValue = 1.f;
        neuralNetwork.criteria.maxEpochs = 25;
        neuralNetwork.criteria.minProgress = 5.0f;
        neuralNetwork.criteria.maxGeneralizationLoss = 4.0f;

        initNeuralNetwork(&neuralNetwork);

        float *expectedOutput =  (float *) malloc (neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1] * sizeof(float));

        loadInputData("cancer.dt", &neuralNetwork, &taskData);
        float generalizationLoss, progress;
        for (neuralNetwork.state.epoch = 0; neuralNetwork.state.epoch < neuralNetwork.criteria.maxEpochs; neuralNetwork.state.epoch++) {
            neuralNetwork.state.learningLine = 0;
            neuralNetwork.state.testLine = 0;
            /**
             * Learning
             */
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["o_network_learning"].Start();
            #endif
            while (getLearningVector(&neuralNetwork, &taskData, expectedOutput)) {
                neuralLearnCycle(&neuralNetwork, expectedOutput);
            }
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["o_network_learning"].Stop();
            #endif

            /**
             * Testing
             */
            neuralNetwork.currentSquareErrorCounter = 0.0f;
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["o_network_testing"].Start();
            #endif
            while (getTestVector(&neuralNetwork, &taskData, expectedOutput)) {
                neuralTestCycle(&neuralNetwork, expectedOutput);
            }
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["o_network_testing"].Stop();
            #endif

            neuralNetwork.squareErrorHistory[neuralNetwork.state.epoch] = (neuralNetwork.setup.maxOutputValue - neuralNetwork.setup.minOutputValue) * neuralNetwork.currentSquareErrorCounter/ (neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1] * taskData.totalTestLines);
            findAndSetBestSquareError(&neuralNetwork);
            // std::cout << neuralNetwork.squareErrorHistory[neuralNetwork.state.epoch] << std::endl;
            if (neuralNetwork.state.epoch > 0) {
                generalizationLoss = neuralNetwork.squareErrorHistory[neuralNetwork.state.epoch] / neuralNetwork.bestSquareError[1] - 1;
                progress = squareErrorSum(&neuralNetwork) / ((neuralNetwork.state.epoch + 1) * neuralNetwork.bestSquareError[1]) - 1;
                if (generalizationLoss > neuralNetwork.criteria.maxGeneralizationLoss || progress < neuralNetwork.criteria.minProgress) {
                    // break;
                }
            }
            // std::cout << neuralNetwork.squareErrorHistory[neuralNetwork.state.epoch] << "  "<< neuralNetwork.bestSquareError[1]<< "\t"<< generalizationLoss << "\t\t"<< progress<< std::endl;
        }
        neuralNetwork.state.epoch--;
        printf("END Learn: %f Epochs: %f Best avg err %f\n", neuralNetwork.setup.learningFactor, neuralNetwork.bestSquareError[0], neuralNetwork.bestSquareError[1]);
        printClassificationAccurancy(&(neuralNetwork.classificationAccurancyHistory[4 * neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1] * neuralNetwork.state.epoch]), neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1]);

        deleteNeuralNetwork(&neuralNetwork);
        deleteTaskData(&taskData);
        free(expectedOutput);
    }
}
