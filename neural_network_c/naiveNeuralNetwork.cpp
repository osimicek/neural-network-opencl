#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include "initPapi.h"
#include "naiveNeuralNetwork.h"

namespace naive {

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
     * Prints accurancy for every output neuron.
     */
    void printClassificationAccurancy(int *accurancy, int numOfNeurons) {
        for (int neuron = 0; neuron < numOfNeurons; neuron++) {
            printf("N. %d\t| true T | true F | precision\n", neuron);
            printf("classification. T\t| %d\t | %d\t  | %5.1f%%\n", accurancy[neuron * 4], accurancy[neuron * 4 + 1], accurancy[neuron * 4] / ((accurancy[neuron * 4] + accurancy[neuron * 4 + 1]) / 100.0));
            printf("classification. F\t| %d\t | %d\t  | %5.1f%%\n", accurancy[neuron * 4 + 2], accurancy[neuron * 4 + 3], accurancy[neuron * 4 + 3] / ((accurancy[neuron * 4 + 2] + accurancy[neuron * 4 + 3]) / 100.0));
            printf("recall\t| %5.1f%% | %5.1f%% \n\n", accurancy[neuron * 4] / ((accurancy[neuron * 4] + accurancy[neuron * 4 + 2]) / 100.0),
                                          accurancy[neuron * 4 + 3] / ((accurancy[neuron * 4 + 1] + accurancy[neuron * 4 + 3]) / 100.0));
        }
    }


    /**
     * Initialization of weights with random values. Seed is fixed to achieve reproducibility.
     */
    void initWeights(float *weights, int length) {
        // srand(37);
        for (int i = 0; i < length; i++) {
            weights[i] = (rand() % 100 + 1) / 100.f;
        }
    }

    /**
     * Initialization accurancy array.
     */
    void initAccurancy(int *accurancy, int length) {
        for (int i = 0; i < length; i++) {
            accurancy[i] = 0;
        }
    }

    /**
     * Finds lowest square error and sets it as the best one.
     */
    void findAndSetBestSquareError(NeuralNetworkT *neuralNetwork) {
        neuralNetwork->bestSquareError[0] = 0;
        neuralNetwork->bestSquareError[1] = neuralNetwork->squareErrorHistory[0];

        for (int i = 1; i <= neuralNetwork->state.epoch; i++) {
            if (neuralNetwork->squareErrorHistory[i] < neuralNetwork->bestSquareError[1]) {
                neuralNetwork->bestSquareError[0] = i;
                neuralNetwork->bestSquareError[1] = neuralNetwork->squareErrorHistory[i];
            }
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
     * Returns a learning vector for neural network.
     */
    bool getLearningVector(NeuralNetworkT *neuralNetwork, TaskData *taskData, float *expectedOutput) {

        if (neuralNetwork->state.learningLine >= taskData->totalLearningLines) {
            return false;
        }
        for (int input = 0; input < neuralNetwork->setup.layers[0]; input++) {
            neuralNetwork->state.values[input] = taskData->learningInputs[input + neuralNetwork->state.learningLine * neuralNetwork->setup.layers[0]];
        }

        for (int output = 0; output < neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]; output++) {
            expectedOutput[output] = taskData->learningOutputs[output + neuralNetwork->state.learningLine * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]];
        }
        neuralNetwork->state.learningLine ++;
        return true;
    }

    /**
     * Returns a test vector for neural network.
     */
    bool getTestVector(NeuralNetworkT *neuralNetwork, TaskData *taskData, float *expectedOutput) {

        if (neuralNetwork->state.testLine >= taskData->totalTestLines) {
            return false;
        }
        for (int input = 0; input < neuralNetwork->setup.layers[0]; input++) {
            neuralNetwork->state.values[input] = taskData->testInputs[input + neuralNetwork->state.testLine * neuralNetwork->setup.layers[0]];
        }

        for (int output = 0; output < neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]; output++) {
            expectedOutput[output] = taskData->testOutputs[output + neuralNetwork->state.testLine * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]];
        }
        neuralNetwork->state.testLine ++;
        return true;
    }

    /**
     * Returns a classification vector for neural network.
     */
    bool getClassificationVector(NeuralNetworkT *neuralNetwork, TaskData *taskData, float **classificationOutput) {

        if (neuralNetwork->state.learningLine >= taskData->totalLearningLines) {
            return false;
        }
        for (int input = 0; input < neuralNetwork->setup.layers[0]; input++) {
            neuralNetwork->state.values[input] = taskData->learningInputs[input + neuralNetwork->state.learningLine * neuralNetwork->setup.layers[0]];
        }

        *classificationOutput = &(taskData->learningOutputs[neuralNetwork->state.learningLine * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]]);
        neuralNetwork->state.learningLine ++;
        return true;
    }

    /**
     * Reads and stores input vectors for learning and testing neural network.
     */
    void loadInputData(const char* filename, NeuralNetworkT *neuralNetwork, TaskData *taskData) {
        std::ifstream input(filename);
        if (input.fail()) {
            fprintf(stderr,"Could not open file %s \n", filename);
            exit(-1);
        }
        
        int inputVectorSize, outputVectorSize, totalLearningLines, totalTestLines;
        input >> inputVectorSize;
        input >> outputVectorSize;
        input >> totalLearningLines;
        input >> totalTestLines;

        neuralNetwork->setup.layers[0] = inputVectorSize;
        neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] = outputVectorSize;
        float value;
        unsigned long int learningInputCounter = 0;
        unsigned long int learningOutputCounter = 0;
        taskData->learningInputs = (float *) malloc (totalLearningLines * inputVectorSize * sizeof(float));
        taskData->learningOutputs = (float *) malloc (totalLearningLines * outputVectorSize * sizeof(float));
        for (unsigned int row = 0; row < totalLearningLines; row++) {
            for (unsigned int inputCol = 0; inputCol < inputVectorSize; inputCol++) {
                input >> value;
                taskData->learningInputs[learningInputCounter] = value;
                learningInputCounter++;
            }

            for (unsigned int outputCol = 0; outputCol < outputVectorSize; outputCol++) {
                input >> value;
                taskData->learningOutputs[learningOutputCounter] = value;
                learningOutputCounter++;
            }
        }

        unsigned long int testInputCounter = 0;
        unsigned long int testOutputCounter = 0;
        taskData->testInputs = (float *) malloc (totalTestLines * inputVectorSize * sizeof(float));
        taskData->testOutputs = (float *) malloc (totalTestLines * outputVectorSize * sizeof(float));
        for (unsigned int row = 0; row < totalTestLines; row++) {
            for (unsigned int inputCol = 0; inputCol < inputVectorSize; inputCol++) {
                input >> value;
                taskData->testInputs[testInputCounter] = value;
                testInputCounter++;
            }

            for (unsigned int outputCol = 0; outputCol < outputVectorSize; outputCol++) {
                input >> value;
                taskData->testOutputs[testOutputCounter] = value;
                testOutputCounter++;
            }
        }

        taskData->totalLearningLines = totalLearningLines;
        taskData->totalTestLines = totalTestLines;
    }

     /**
     * Reads and stores input vectors for neural network classification.
     */
    void loadClassificationData(const char* filename, NeuralNetworkT *neuralNetwork, TaskData *taskData) {
        std::ifstream input(filename);
        if (input.fail()) {
            fprintf(stderr,"Could not open file %s \n", filename);
            exit(-1);
        }
        
        int inputVectorSize, outputVectorSize, totalLines, totalTestLines;
        input >> inputVectorSize;
        input >> outputVectorSize;
        input >> totalLines;

        neuralNetwork->setup.layers[0] = inputVectorSize;
        neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] = outputVectorSize;
        float value;
        unsigned long int learningInputCounter = 0;
        taskData->learningInputs = (float *) malloc (totalLines * inputVectorSize * sizeof(float));
        // std::cout << filename<<" " <<inputVectorSize << " " << outputVectorSize << " " << totalLines << std::endl;
        taskData->learningOutputs = (float *) malloc (totalLines * outputVectorSize * sizeof(float));
        for (unsigned int row = 0; row < totalLines; row++) {
            for (unsigned int inputCol = 0; inputCol < inputVectorSize; inputCol++) {
                input >> value;
                taskData->learningInputs[learningInputCounter] = value;
                learningInputCounter++;
            }
        }
        taskData->totalLearningLines = totalLines;
        input.close();
    }


    /**
     * Stores output vectors for neural network classification.
     */
    void storeClassification(const char* filename, NeuralNetworkT *neuralNetwork, TaskData *taskData) {
        std::ofstream out(filename);

        for (unsigned int row = 0; row < taskData->totalLearningLines; row++) {
            int numOfOutputNeurons = neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1];
            for (int output = 0; output < numOfOutputNeurons; output++) {
                out << taskData->learningOutputs[output + row * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]];
                if (output != numOfOutputNeurons -1) {
                    out << " ";
                }
            }
            out << std::endl;
        }
        out.close();
    }

    /**
     * Initialization of neural network
     */
    void initNeuralNetwork(NeuralNetworkT *neuralNetwork) {
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
     * Frees alocated memory
     */
    void deleteNeuralNetwork(NeuralNetworkT *neuralNetwork) {
        free(neuralNetwork->state.weights);
        free(neuralNetwork->state.values);
        free(neuralNetwork->state.errors);
        if (neuralNetwork->setup.classification) {
            free(neuralNetwork->classificationAccurancyHistory);
        }
        free(neuralNetwork->squareErrorHistory);
    }

    /**
     * Frees alocated memory
     */
    void deleteTaskData(TaskData *taskData) {
        free(taskData->learningInputs);
        free(taskData->learningOutputs);
        free(taskData->testInputs);
        free(taskData->testOutputs);
    }

    /**
     * Performs a learning cycle of neural network. Input vector is transformed to output vector.
     * Output vector si compared with expected output and error is backpropagated throw net and
     * weights are modified.
     */
    void neuralLearnCycle(NeuralNetworkT *neuralNetwork, 
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
        (*papi_routines)["network_learning_foward_computation"].Start();
        #endif
        for (int layer = 1; layer < numOfLayers; layer++) {
            int neurons = layers[layer];
            int prevNeurons = layers[layer - 1];
            valueOffset += prevNeurons;
            for (int neuron = 0; neuron < neurons; neuron++) {
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["network_learning_neural_row_foward_computation"].Start();
                }
                #endif
                float value = 0;
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    value += values[prevNeuron + prevValueOffset] * weights[prevNeuron + weightOffset];
                }
                // std::cout << value << "  ";
                values[neuron + valueOffset] = 1. / (1. + exp(-neuralNetwork->setup.lambda * value));
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["network_learning_neural_row_foward_computation"].Stop();
                }
                #endif
                weightOffset += prevNeurons;
            }
            prevValueOffset += prevNeurons;
        }
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["network_learning_foward_computation"].Stop();
        #endif

        // backwards error computation
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["network_learning_error_computation"].Start();
        #endif
        for (int neuron = 0; neuron < layers[numOfLayers - 1]; neuron++) {
            #if USE_PAPI_NEURAL_ROW_LEARN
            if (neuron == 0) {
                (*papi_routines)["network_learning_neural_row_error_computation"].Start();
            }
            #endif
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
            #if USE_PAPI_NEURAL_ROW_LEARN
            if (neuron == 0) {
                (*papi_routines)["network_learning_neural_row_error_computation"].Stop();
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
                    (*papi_routines)["network_learning_neural_row_error_computation"].Start();
                }
                #endif
                float weightError = 0.f;
                for (int followNeuron = 0; followNeuron < followNeurons; followNeuron++) {
                    weightError += errors[followNeuron + followValueOffset] *
                                  weights[followNeuron + weightOffset];
                }
                weightOffset += followNeurons;
                float value = values[neuron + valueOffset];
                float ex = exp(-neuralNetwork->setup.lambda*value);
                errors[neuron + valueOffset] = weightError * (ex * (1 - ex));
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["network_learning_neural_row_error_computation"].Stop();
                }
                #endif
            }
            weightOffset -= neurons * followNeurons;
            followValueOffset -= neurons;
        }
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["network_learning_error_computation"].Stop();
        #endif
        // printNeuralNetwork(neuralNetwork, expectedOutput);
        // error propagation
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["network_learning_weight_update"].Start();
        #endif
        weightOffset = 0;
        valueOffset = 0;
        for (int layer = 1; layer < numOfLayers; layer++) {
            int neurons = layers[layer];
            int prevNeurons = layers[layer - 1];
            for (int neuron = 0; neuron < neurons; neuron++) {
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["network_learning_neural_row_weight_update"].Start();
                }
                #endif
                float value = 0.f;
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    weights[prevNeuron + weightOffset] = weights[prevNeuron + weightOffset] +
                                                         neuralNetwork->setup.learningFactor * values[prevNeuron + valueOffset] *
                                                         errors[neuron + valueOffset + prevNeurons];
                }
                #if USE_PAPI_NEURAL_ROW_LEARN
                if (neuron == 0) {
                    (*papi_routines)["network_learning_neural_row_weight_update"].Stop();
                }
                #endif
                weightOffset += prevNeurons;
            }
            valueOffset += prevNeurons;
        }
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["network_learning_weight_update"].Stop();
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
        int *layers = neuralNetwork->setup.layers;
        float *values = neuralNetwork->state.values;
        float *weights = neuralNetwork->state.weights;

        int valueOffset = 0;
        int prevValueOffset = 0;
        int weightOffset = 0;

        // foward value computation
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["network_testing_foward_computation"].Start();
        #endif
        for (int layer = 1; layer < numOfLayers; layer++) {
            int neurons = layers[layer];
            int prevNeurons = layers[layer - 1];
            valueOffset += prevNeurons;
            for (int neuron = 0; neuron < neurons; neuron++) {
                #if USE_PAPI_NEURAL_ROW_TEST
                if (neuron == 0) {
                    (*papi_routines)["network_testing_neural_row_foward_computation"].Start();
                }
                #endif
                float value = 0.f;
                for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                    value += values[prevNeuron + prevValueOffset] * weights[prevNeuron + weightOffset];
                }
                // std::cout << value << "  ";
                values[neuron + valueOffset] = 1.f / (1.f + exp(-neuralNetwork->setup.lambda * value));
                #if USE_PAPI_NEURAL_ROW_TEST
                if (neuron == 0) {
                    (*papi_routines)["network_testing_neural_row_foward_computation"].Stop();
                }
                #endif
                weightOffset += prevNeurons;
            }
            prevValueOffset += prevNeurons;
        }

        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["network_testing_foward_computation"].Stop();
        #endif


        // error computation backwards
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["network_testing_error_computation"].Start();
        #endif
        int numOfOutputNeurons = layers[numOfLayers - 1];
        for (int neuron = 0; neuron < numOfOutputNeurons; neuron++) {
            #if USE_PAPI_NEURAL_ROW_TEST
            if (neuron == 0) {
                (*papi_routines)["network_testing_neural_row_error_computation"].Start();
            }
            #endif
            float value = values[neuron + valueOffset];
            float diff;
            

            if (neuralNetwork->setup.classification) {
                // conversion to bool
                int classValue = round(value);
                // neuralNetwork->currentSquareErrorCounter += diff * diff; 

                int pos = 4 * neuron + 4 * numOfOutputNeurons * neuralNetwork->state.epoch;
                if (classValue == 1 && expectedOutput[neuron] == 1.f) {
                    // true positive
                    neuralNetwork->classificationAccurancyHistory[pos] += 1;
                } else if (classValue == 1 && expectedOutput[neuron] == 0.f) {
                    // true negative
                    // diff = expectedOutput[neuron] - value;
                    neuralNetwork->currentSquareErrorCounter += 1; 

                    neuralNetwork->classificationAccurancyHistory[pos + 1] += 1;
                } else if (classValue == 0 && expectedOutput[neuron] == 1.f) {
                    // false negative
                    // diff = expectedOutput[neuron] - value;
                    neuralNetwork->currentSquareErrorCounter += 1; 

                    neuralNetwork->classificationAccurancyHistory[pos + 2] += 1;
                } else if (classValue == 0 && expectedOutput[neuron] == 0.f) {
                    // false positive
                    neuralNetwork->classificationAccurancyHistory[pos + 3] += 1; 
                }
            } else {
                diff = expectedOutput[neuron] - value;
                neuralNetwork->currentSquareErrorCounter += diff * diff; 
            }
            #if USE_PAPI_NEURAL_ROW_TEST
            if (neuron == 0) {
                (*papi_routines)["network_testing_neural_row_error_computation"].Stop();
            }
            #endif
        }
        #if USE_PAPI_LEARN_DETAIL
        (*papi_routines)["network_testing_error_computation"].Stop();
        #endif 
    }

    /**
     * Performs trainign and testing of neural network.
     */
    void runNaiveNeuralNetwork(NeuralNetworkT *neuralNetwork, const char* taskFilename) {
        TaskData taskData;

        loadInputData(taskFilename, neuralNetwork, &taskData);
        initNeuralNetwork(neuralNetwork);

        float *expectedOutput =  (float *) malloc (neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * sizeof(float));

        
        float generalizationLoss, progress;
        for (neuralNetwork->state.epoch = 0; neuralNetwork->state.epoch < neuralNetwork->criteria.maxEpochs; neuralNetwork->state.epoch++) {
            neuralNetwork->state.learningLine = 0;
            neuralNetwork->state.testLine = 0;
            /**
             * Learning
             */
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["network_learning"].Start();
            #endif
            int counter = 0;
            while (getLearningVector(neuralNetwork, &taskData, expectedOutput)) {
                neuralLearnCycle(neuralNetwork, expectedOutput);
                if (counter > 20) {
                    // return;
                }
                counter ++;
            }
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["network_learning"].Stop();
            #endif

            /**
             * Testing
             */
            neuralNetwork->currentSquareErrorCounter = 0.0f;
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["network_testing"].Start();
            #endif
            while (getTestVector(neuralNetwork, &taskData, expectedOutput)) {
                neuralTestCycle(neuralNetwork, expectedOutput);
            }
            #if USE_PAPI_LEARN_AND_TEST
            (*papi_routines)["network_testing"].Stop();
            #endif

            neuralNetwork->squareErrorHistory[neuralNetwork->state.epoch] = (neuralNetwork->setup.maxOutputValue - neuralNetwork->setup.minOutputValue) * neuralNetwork->currentSquareErrorCounter/ (neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * taskData.totalTestLines);
            findAndSetBestSquareError(neuralNetwork);
            // std::cout << neuralTestCycleNetwork.squareErrorHistory[neuralNetwork->state.epoch] << std::endl;
            if (neuralNetwork->state.epoch > 0) {
                generalizationLoss = neuralNetwork->squareErrorHistory[neuralNetwork->state.epoch] / neuralNetwork->bestSquareError[1] - 1;
                progress = squareErrorSum(neuralNetwork) / ((neuralNetwork->state.epoch + 1) * neuralNetwork->bestSquareError[1]) - 1;
                if (generalizationLoss > neuralNetwork->criteria.maxGeneralizationLoss || progress < neuralNetwork->criteria.minProgress) {
                    // break;
                }
            }
            // std::cout << neuralNetwork->squareErrorHistory[neuralNetwork->state.epoch] << "  "<< neuralNetwork->bestSquareError[1]<< "\t"<< generalizationLoss << "\t\t"<< progress<< std::endl;
        }
        neuralNetwork->state.epoch--;
        printf("NN RESULT: Best avg err: %f Epochs: %f Learn factor: %f \n", neuralNetwork->bestSquareError[1], neuralNetwork->bestSquareError[0], neuralNetwork->setup.learningFactor);
        printClassificationAccurancy(&(neuralNetwork->classificationAccurancyHistory[4 * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * neuralNetwork->state.epoch]), neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]);

        deleteNeuralNetwork(neuralNetwork);
        deleteTaskData(&taskData);
        free(expectedOutput);
    }
}
