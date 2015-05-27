#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <mm_malloc.h>
#include "initPapi.h"
#include "optimizedNeuralNetwork.h"
using namespace naive;

#define SHOW_CLASSIFICATION_ACCURANCY 0

#define MEMORY_ALIGN 64


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

    void printNeuralNetwork(NeuralNetworkT *neuralNetwork, float *expectedOutput) {
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
                    printf("%0.2f ",  weights[weightOffset + prevNeuron]);
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
     * Initialization of weights with random values. Seed is fixed to achieve reproducibility.
     */
    void initWeights(float *weights, int length) {
        for (int i = 0; i < length; i++) {
            weights[i] = (rand() % 100 + 1) / 100.f - 0.5;
        }
    }



    /**
     * Returns sum of squareErrors
     */
    float squareErrorSum(NeuralNetworkT *neuralNetwork) {
        float sum = 0.0f;
        for (int i = 0; i <= neuralNetwork->state.epoch; i++) {
            sum += neuralNetwork->squareErrorHistory[i];
        }
        return sum;
    }




    /**
     * Initialization of neural network
     */
    void initNeuralNetwork(NeuralNetworkT *neuralNetwork) {
        int numOfWeights = 0;
        for (int i = 1; i < neuralNetwork->setup.numOfLayers; i++) {
            numOfWeights += neuralNetwork->setup.layers[i] * neuralNetwork->setup.layers[i - 1];
        }

        neuralNetwork->state.weights = (float *) _mm_malloc(numOfWeights * sizeof(float), MEMORY_ALIGN);
        initWeights(neuralNetwork->state.weights, numOfWeights);

        #if SHOW_CLASSIFICATION_ACCURANCY
        int classificationAccurancyLength = 4 * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * neuralNetwork->criteria.maxEpochs;
        if (neuralNetwork->setup.classification) {
            neuralNetwork->classificationAccurancyHistory = (int *) _mm_malloc(classificationAccurancyLength * sizeof(int), MEMORY_ALIGN);
        }
        initAccurancy(neuralNetwork->classificationAccurancyHistory, classificationAccurancyLength);
        #endif
        neuralNetwork->squareErrorHistory = (float *) _mm_malloc(2 * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * neuralNetwork->criteria.maxEpochs * sizeof(int), MEMORY_ALIGN);
        neuralNetwork->bestSquareError[0] = -1.f;
        neuralNetwork->bestSquareError[1] = -1.f;
        neuralNetwork->currentSquareErrorCounter = 0.0f;

        int numOfValues = 0;
        for (int i = 0; i < neuralNetwork->setup.numOfLayers; i++) {
            numOfValues += neuralNetwork->setup.layers[i];
        }
        neuralNetwork->state.values =  (float *) _mm_malloc(numOfValues * sizeof(float), MEMORY_ALIGN);
        neuralNetwork->state.errors =  (float *) _mm_malloc(numOfValues * sizeof(float), MEMORY_ALIGN);
    }

    /**
     * Frees alocated memory
     */
    void deleteNeuralNetwork(NeuralNetworkT *neuralNetwork) {
        free(neuralNetwork->state.weights);
        free(neuralNetwork->state.values);
        free(neuralNetwork->state.errors);
        #if SHOW_CLASSIFICATION_ACCURANCY
        if (neuralNetwork->setup.classification) {
            free(neuralNetwork->classificationAccurancyHistory);
        }
        #endif
        free(neuralNetwork->squareErrorHistory);
    }


    /**
     * Performs a learning cycle of neural network. Input vector is transformed to output vector.
     * Output vector si compared with expected output and error is backpropagated throw net and
     * weights are modified.
     */
    void neuralLearnCycle(NeuralNetworkT *neuralNetwork, 
                    float *expectedOutput) {
        int numOfLayers = neuralNetwork->setup.numOfLayers;
        int * __restrict layers = neuralNetwork->setup.layers;
        float * __restrict values = neuralNetwork->state.values;
        float * __restrict weights = neuralNetwork->state.weights;
        float * __restrict errors = neuralNetwork->state.errors;


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
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["o_network_learning_neural_row_foward_computation"].Start();
                }
                #endif
                float value = 0;
                #pragma ivdep
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    value += values[prevNeuron + prevValueOffset] * weights[prevNeuron + weightOffset];
                    // std::cout << "waatch " << prevNeuron + prevValueOffset <<" * " << prevNeuron + weightOffset << "      "<< prevValueOffset<< std::endl;
                }
                // std::cout << value << "  ";
                values[neuron + valueOffset] = 1. / (1. + exp(-value));
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["o_network_learning_neural_row_foward_computation"].Stop();
                }
                #endif
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
            #if USE_PAPI_NEURAL_ROW_LEARN
            if (neuron == 0) {
                (*papi_routines)["o_network_learning_neural_row_error_computation"].Start();
            }
            #endif
            float value = values[neuron + valueOffset];
            float cmpValue = value;
            if (neuralNetwork->setup.classification) {
                cmpValue = round(value);
            }
            // std::cout << value << " ";

            // if (expectedOutput[neuron] - cmpValue) {
                // std::cout << "chyba u " << neuron << " " << value << std::endl;
                // float ex = 1.f / (1.f + exp(-neuralNetwork->setup.lambda * value));;
                // std::cout << "velikost ch " << (expectedOutput[neuron] - value) * (ex * (1 - ex)) << " ex " << ex << std::endl;
                errors[neuron + valueOffset] = (expectedOutput[neuron] - value) * (value * (1 - value));
            // } else {
                // errors[neuron + valueOffset] = 0.f;
            // }
            #if USE_PAPI_NEURAL_ROW_LEARN
            if (neuron == 0) {
                (*papi_routines)["o_network_learning_neural_row_error_computation"].Stop();
            }
            #endif
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
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["o_network_learning_neural_row_error_computation"].Start();
                }
                #endif
                float weightError = 0.f;
                #pragma ivdep
                for (int followNeuron = 0; followNeuron < followNeurons; followNeuron++) {
                    weightError += errors[followNeuron + followValueOffset] *
                                  weights[neuron + followNeuron * neurons + weightOffset];
                    // std::cout << "waatch " <<followValueOffset<< " "<< neuron + weightOffset + followNeuron * neurons << "       " << errors[followNeuron + followValueOffset] <<" * " <<weights[neuron + followNeuron * neurons + weightOffset]<< " = " << weightError<< std::endl;
                }
                // je potreba razantne snizit vahy prvni vrstvy, proce se tak nedeje samo?

                float value = values[neuron + valueOffset];
                // float ex = exp(-value);
                errors[neuron + valueOffset] = weightError * (value * (1 - value));
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["o_network_learning_neural_row_error_computation"].Stop();
                }
                #endif
            }
            followValueOffset -= neurons;
        }
        // return;
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
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["o_network_learning_neural_row_weight_update"].Start();
                }
                #endif
                float value = 0.f;
                #pragma ivdep
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    weights[prevNeuron + weightOffset] = weights[prevNeuron + weightOffset] +
                                                         neuralNetwork->setup.learningFactor * values[prevNeuron + valueOffset] *
                                                         errors[neuron + valueOffset + prevNeurons];
                    // std::cout << "waatch " << prevNeuron + weightOffset << "      "<< prevNeuron + valueOffset << "  "<< neuron + valueOffset + prevNeurons<< std::endl;
                }
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["o_network_learning_neural_row_weight_update"].Stop();
                }
                #endif
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
    void neuralTestCycle(NeuralNetworkT *neuralNetwork, 
                    float *expectedOutput) {
        
        int numOfLayers = neuralNetwork->setup.numOfLayers;
        int * __restrict layers = neuralNetwork->setup.layers;
        float * __restrict values = neuralNetwork->state.values;
        float * __restrict weights = neuralNetwork->state.weights;

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
                #if USE_PAPI_NEURAL_ROW_TEST
                if (neuron == 0) {
                    (*papi_routines)["o_network_testing_neural_row_foward_computation"].Start();
                }
                #endif
                float value = 0.f;
                #pragma ivdep
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    value += values[prevNeuron + prevValueOffset] * weights[prevNeuron + weightOffset];
                    // std::cout << "waatch " << prevNeuron + prevValueOffset <<" * " << prevNeuron + weightOffset << "      "<< prevValueOffset<< std::endl;
                }
                // std::cout << value << "  ";
                values[neuron + valueOffset] = 1.f / (1.f + exp(-value));
                #if USE_PAPI_NEURAL_ROW_TEST
                if (neuron == 0) {
                    (*papi_routines)["o_network_testing_neural_row_foward_computation"].Stop();
                }
                #endif
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
        float classificationCandidateValue = -1.f;
        int classificationCandidateIndex = 0;
        for (int neuron = 0; neuron < numOfOutputNeurons; neuron++) {
            #if USE_PAPI_NEURAL_ROW_TEST
            if (neuron == 0) {
                (*papi_routines)["o_network_testing_neural_row_error_computation"].Start();
            }
            #endif
            float value = values[neuron + valueOffset];
            float diff;
            
            if (neuralNetwork->setup.classification) {
                // conversion to bool
                int classValue = round(value);

                if (classValue == 1 && classificationCandidateValue < value) {
                    classificationCandidateValue = value;
                    classificationCandidateIndex = neuron;
                }
            } else {
                diff = expectedOutput[neuron] - value;
                neuralNetwork->currentSquareErrorCounter += diff * diff; 
            }
            #if USE_PAPI_NEURAL_ROW_TEST
            if (neuron == 0) {
                (*papi_routines)["o_network_testing_neural_row_error_computation"].Stop();
            }
            #endif
            
        }
        if (neuralNetwork->setup.classification) {
            if (expectedOutput[classificationCandidateIndex] != 1.f || classificationCandidateValue == -1.f) {
                neuralNetwork->currentSquareErrorCounter += 1;
            }
        }
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["o_network_testing_error_computation"].Stop();
        #endif 
    }


    /**
     * Performs a classification cycle of neural network. Input vector is transformed to output vector.
     * Output vector si stored in suplided data array.
     */
    void neuralClassificationCycle(NeuralNetworkT *neuralNetwork, 
                        float *classificationOutput) {
        
        int numOfLayers = neuralNetwork->setup.numOfLayers;
        int * __restrict layers = neuralNetwork->setup.layers;
        float * __restrict values = neuralNetwork->state.values;
        float * __restrict weights = neuralNetwork->state.weights;

        int valueOffset = 0;
        int prevValueOffset = 0;
        int weightOffset = 0;

        // foward value computation
        for (int layer = 1; layer < numOfLayers; layer++) {
            int neurons = layers[layer];
            int prevNeurons = layers[layer - 1];
            valueOffset += prevNeurons;
            for (int neuron = 0; neuron < neurons; neuron++) {
                float value = 0.f;
                #pragma ivdep
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    value += values[prevNeuron + prevValueOffset] * weights[prevNeuron + weightOffset];
                    // std::cout << "waatch " << prevNeuron + prevValueOffset <<" * " << prevNeuron + weightOffset << "      "<< prevValueOffset<< std::endl;
                }
                // std::cout << value << "  ";
                values[neuron + valueOffset] = 1.f / (1.f + exp(-value));
                weightOffset += prevNeurons;
            }
            prevValueOffset += prevNeurons;
        }

        // error computation backwards
        int numOfOutputNeurons = layers[numOfLayers - 1];
        float classificationCandidateValue = -1.f;
        int classificationCandidateIndex = 0;
        for (int neuron = 0; neuron < numOfOutputNeurons; neuron++) {
            float value = values[neuron + valueOffset];
            float diff;
                // conversion to bool
                int classValue = round(value);
                classificationOutput[neuron] = 0.f;

                if (classValue == 1 && classificationCandidateValue < value) {
                    classificationCandidateValue = value;
                    classificationCandidateIndex = neuron;
                }
            
        }
        classificationOutput[classificationCandidateIndex] = 1.f;
    }

    /**
     * Performs trainign and testing of neural network.
     */
    void runOptimizedNeuralNetwork(NeuralNetworkT *neuralNetwork, const char* taskFilename, bool verbose) {
        TaskData taskData;

        loadInputData(taskFilename, neuralNetwork, &taskData);
        initNeuralNetwork(neuralNetwork);

        float *expectedOutput =  (float *) _mm_malloc(neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * sizeof(float), MEMORY_ALIGN);

        
        float generalizationLoss, progress;
        if (verbose) {
            std::cout << " STARTING TRAINING:" << std::endl;
        }
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        for (neuralNetwork->state.epoch = 0; neuralNetwork->state.epoch < neuralNetwork->criteria.maxEpochs; neuralNetwork->state.epoch++) {
            neuralNetwork->state.learningLine = 0;
            neuralNetwork->state.testLine = 0;
            /**
             * Learning
             */
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["o_network_learning"].Start();
            #endif
            int counter = 0;
            while (getLearningVector(neuralNetwork, &taskData, expectedOutput)) {
                neuralLearnCycle(neuralNetwork, expectedOutput);
                // if (counter > 10) {
                //     #if USE_PAPI_LEARN_AND_TEST
                //     (*papi_routines)["o_network_learning"].Stop();
                //     #endif
                //     return;
                // }
                // counter ++;
                // if (neuralNetwork->state.learningLine >= 15 && neuralNetwork->state.epoch >= 0) {
                //     printNeuralNetwork(neuralNetwork, expectedOutput);
                //     std::cout << "currentSquareErrorCounter " << neuralNetwork->currentSquareErrorCounter<<std::endl;
                //     return;
                // }
            }
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["o_network_learning"].Stop();
            #endif

            /**
             * Testing
             */
            neuralNetwork->currentSquareErrorCounter = 0.0f;
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["o_network_testing"].Start();
            #endif
            while (getTestVector(neuralNetwork, &taskData, expectedOutput)) {
                neuralTestCycle(neuralNetwork, expectedOutput);
                // if (neuralNetwork->state.testLine > 1 && neuralNetwork->state.epoch >= 0) {
                //     printNeuralNetwork(neuralNetwork, expectedOutput);
                //     std::cout << "currentSquareErrorCounter " << neuralNetwork->currentSquareErrorCounter<<std::endl;
                //     return;
                // }
            }
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["o_network_testing"].Stop();
            #endif

            neuralNetwork->squareErrorHistory[neuralNetwork->state.epoch] = (neuralNetwork->setup.maxOutputValue - neuralNetwork->setup.minOutputValue) * neuralNetwork->currentSquareErrorCounter/ (neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * taskData.totalTestLines);
            findAndSetBestSquareError(neuralNetwork);
            // std::cout << neuralNetwork->squareErrorHistory[neuralNetwork->state.epoch] << std::endl;
            // if (neuralNetwork->state.epoch > 0) {
            //     generalizationLoss = neuralNetwork->squareErrorHistory[neuralNetwork->state.epoch] / neuralNetwork->bestSquareError[1] - 1;
            //     progress = squareErrorSum(neuralNetwork) / ((neuralNetwork->state.epoch + 1) * neuralNetwork->bestSquareError[1]) - 1;
            //     if (generalizationLoss > neuralNetwork->criteria.maxGeneralizationLoss || progress < neuralNetwork->criteria.minProgress) {
            //         // break;
            //     }
            // }
            // std::cout << neuralNetwork->squareErrorHistory[neuralNetwork->state.epoch] << "  "<< neuralNetwork->bestSquareError[1]<< "\t"<< generalizationLoss << "\t\t"<< progress<< std::endl;
        }
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        if (verbose) {
            std::cout << " Duration: " << (std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() ) / 1000000. << "s" << std::endl;
        }
        // float x[2];
        // printNeuralNetwork(neuralNetwork, x);
        neuralNetwork->state.epoch--;
        // printf("NN RESULT: Best avg err: %f Epoch: %f Learn factor: %f \n", neuralNetwork->bestSquareError[1], neuralNetwork->bestSquareError[0], neuralNetwork->setup.learningFactor);
        
        #if SHOW_CLASSIFICATION_ACCURANCY
        printClassificationAccurancy(&(neuralNetwork->classificationAccurancyHistory[4 * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * neuralNetwork->state.epoch]), neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]);
        #endif
        // std::cout << "ccurrentSquareErrorCounter " << neuralNetwork->currentSquareErrorCounter<<std::endl;
        // for (int i = 0; i < neuralNetwork->criteria.maxEpochs; i++) {
        //     std::cout << neuralNetwork->squareErrorHistory[i] << " ";
        // }
        // std::cout<< std::endl;

        // deleteNeuralNetwork(neuralNetwork);
        // deleteTaskData(&taskData);
        free(expectedOutput);
    }


    /**
     * Performs classification using neural network.
     */
    void runNeuralNetworkClassification(NeuralNetworkT *neuralNetwork, const char* taskFilename, bool verbose, float** classificationResult) {
        TaskData taskData;

        loadClassificationData(taskFilename, neuralNetwork, &taskData);
        float *classificationOutput;
        
        float generalizationLoss, progress;
        wait();
        if (verbose) {
            std::cout << " STARTING PREDICTION:" << std::endl;
        }
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        neuralNetwork->state.learningLine = 0;
        /**
         * Classification
         */
        while (getClassificationVector(neuralNetwork, &taskData, &classificationOutput)) {
            neuralClassificationCycle(neuralNetwork, classificationOutput);
            // if (counter > 10) {
            //     #if USE_PAPI_LEARN_AND_TEST
            //     (*papi_routines)["o_network_learning"].Stop();
            //     #endif
            //     return;
            // }
            // counter ++;
            // if (neuralNetwork->state.learningLine >= 15 && neuralNetwork->state.epoch >= 0) {
            //     printNeuralNetwork(neuralNetwork, expectedOutput);
            //     std::cout << "currentSquareErrorCounter " << neuralNetwork->currentSquareErrorCounter<<std::endl;
            //     return;
            // }
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        if (verbose) {
            std::cout << "  Duration: " << (std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() ) / 1000000. << "s" << std::endl;
        }

        if (classificationResult != NULL) {
            *classificationResult = taskData.learningOutputs;
        }
        // storeClassification("out.txt", neuralNetwork, &taskData);
    }
}
