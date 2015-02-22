#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <papi.h>
#include "papi_cntr.h"

#define USE_PAPI 1

struct TaskData {
   float *learningInputs, *learningOutputs;
   int learningLine, totalLearningLines;
   float *testInputs, *testOutputs;
   int testLine, totalTestLines;
};

#if USE_PAPI
PapiCounterList papi_routines;
#endif

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

void printNeuralNetwork(float *values,
                        float *weights,
                        float *expectedOutput,
                        float *errors,
                        int *layers,
                        int numOfLayers) {
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
void printAccurancy(int *accurancy, int numOfRecords) {
    for (int neuron = 0; neuron < numOfRecords; neuron += 4) {
        printf("N. %d\t| true T | true F | precision\n", neuron / 4);
        printf("pred. T\t| %d\t | %d\t  | %5.1f%%\n", accurancy[neuron], accurancy[neuron + 1], accurancy[neuron] / ((accurancy[neuron] + accurancy[neuron + 1]) / 100.0));
        printf("pred. F\t| %d\t | %d\t  | %5.1f%%\n", accurancy[neuron + 2], accurancy[neuron + 3], accurancy[neuron + 3] / ((accurancy[neuron + 2] + accurancy[neuron + 3]) / 100.0));
        printf("recall\t| %5.1f%% | %5.1f%% \n\n", accurancy[neuron] / ((accurancy[neuron] + accurancy[neuron + 2]) / 100.0),
                                      accurancy[neuron + 3] / ((accurancy[neuron + 1] + accurancy[neuron + 3]) / 100.0));
    }
}

/**
 * Performs a learning cycle of neural network. Input vector is transformed to output vector.
 * Output vector si compared with expected output and error is backpropagated throw net and
 * weights are modified.
 */
void neuralLearnCycle(float *values,
                float *weights,
                float *expectedOutput,
                float *errors,
                int *layers,
                int numOfLayers,
                float learnFactor,
                float lambda) {
    #if USE_PAPI
    papi_routines["network_learning_cycle"].Start();
    #endif
    int valueOffset = 0;
    int prevValueOffset = 0;
    int weightOffset = 0;

    // foward value counting
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
            values[neuron + valueOffset] = 1. / (1. + exp(-lambda * value));
            weightOffset += prevNeurons;
        }
        prevValueOffset += prevNeurons;
    }

    // error counting backwards
    for (int neuron = 0; neuron < layers[numOfLayers - 1]; neuron++) {
        float value = values[neuron + valueOffset];
        int boolValue;
        if (value > 0.5) { // conversion to bool
            boolValue = 1;
        } else {
            boolValue = 0;
        }

        if (expectedOutput[neuron] - boolValue) {
            // std::cout << "chyba u " << neuron << " " << value << std::endl;
            float ex = 1. / (1. + exp(-lambda * value));;
            // std::cout << "velikost ch " << (expectedOutput[neuron] - value) * (ex * (1 - ex)) << " ex " << ex << std::endl;
            errors[neuron + valueOffset] = (expectedOutput[neuron] - value) * (ex * (1 - ex));
        } else {
            errors[neuron + valueOffset] = 0;
        }
        // std::cout << (expectedOutput[neuron] - value) << std::endl;
    }
    int followValueOffset = valueOffset;
    for (int layer = numOfLayers - 2; layer > 0; layer--) {
        int neurons = layers[layer];
        int followNeurons = layers[layer + 1];
        valueOffset -= neurons;
        // std::cout << "---" << std::endl;
        for (int neuron = neurons - 1; neuron >= 0; neuron--) {
            float weightError = 0;
            weightOffset -= followNeurons;
            for (int followNeuron = 0; followNeuron < followNeurons; followNeuron++) {
                weightError += errors[followNeuron + followValueOffset] *
                              weights[followNeuron + weightOffset];
            }
            float value = values[neuron + valueOffset];
            float ex = exp(-lambda*value);
            errors[neuron + valueOffset] = weightError * (ex * (1 - ex));
        }
        followValueOffset -= neurons;
    }
    // printNeuralNetwork(values, weights, expectedOutput, errors, layers, numOfLayers);
    // error propagation
    weightOffset = 0;
    valueOffset = 0;
    for (int layer = 1; layer < numOfLayers; layer++) {
        int neurons = layers[layer];
        int prevNeurons = layers[layer - 1];
        for (int neuron = 0; neuron < neurons; neuron++) {
            float value = 0;
            for (int prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                weights[prevNeuron + weightOffset] = weights[prevNeuron + weightOffset] +
                                                     learnFactor * values[prevNeuron + valueOffset] *
                                                     errors[neuron + valueOffset + prevNeurons];
            }
            weightOffset += prevNeurons;
        }
        valueOffset += prevNeurons;
    }

    // printNeuralNetwork(values, weights, expectedOutput, errors, layers, numOfLayers);

    #if USE_PAPI
    papi_routines["network_learning_cycle"].Stop();
    #endif
}

/**
 * Performs a test cycle of neural network. Input vector is transformed to output vector.
 * Output vector si compared with expected output. Accurancy is counted from this error.
 */
void neuralTestCycle(float *values,
                float *weights,
                float *expectedOutput,
                int *layers,
                int numOfLayers,
                float lambda,
                int *accurancy) {
    #if USE_PAPI
    papi_routines["network_test_cycle"].Start();
    #endif
    int valueOffset = 0;
    int prevValueOffset = 0;
    int weightOffset = 0;

    // foward value counting
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
            values[neuron + valueOffset] = 1. / (1. + exp(-lambda * value));
            weightOffset += prevNeurons;
        }
        prevValueOffset += prevNeurons;
    }

    // error counting backwards
    for (int neuron = 0; neuron < layers[numOfLayers - 1]; neuron++) {
        float value = values[neuron + valueOffset];
        int boolValue;
        if (value > 0.5) { // conversion to bool
            boolValue = 1;
        } else {
            boolValue = 0;
        }

        if (boolValue == 1 && expectedOutput[neuron] == 1) {
            // true positive
            accurancy[4 * neuron] += 1;
            // std::cout << " ted "<< " " << 4 * neuron <<" "<< accurancy[4 * neuron] << std::endl;
        } else if (boolValue == 1 && expectedOutput[neuron] == 0) {
            // true negative
            accurancy[4 * neuron + 1] += 1;
        } else if (boolValue == 0 && expectedOutput[neuron] == 1) {
            // false negative
            accurancy[4 * neuron + 2] += 1;
        } else if (boolValue == 0 && expectedOutput[neuron] == 0) {
            // false positive
            accurancy[4 * neuron + 3] += 1; 
        }
    }

    #if USE_PAPI
    papi_routines["network_test_cycle"].Stop();
    #endif
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
 * Initialization accurancy array.
 */
void initAccurancy(int *accurancy, int length) {
    for (int i = 0; i < length; i++) {
        accurancy[i] = 0;
    }
}

/**
 * Returns a learning vector for neural network.
 */
bool getLearningVector(TaskData *taskData, int *layers, int numOfLayers, float *values, float *expectedOutput) {
    if (taskData->learningLine >= taskData->totalLearningLines) {
        return false;
    }
    for (int input = 0; input < layers[0]; input++) {
        values[input] = taskData->learningInputs[input + taskData->learningLine * layers[0]];
    }

    for (int output = 0; output < layers[numOfLayers - 1]; output++) {
        expectedOutput[output] = taskData->learningOutputs[output + taskData->learningLine * layers[numOfLayers - 1]];
    }
    taskData->learningLine ++;
    return true;
}

/**
 * Returns a test vector for neural network.
 */
bool getTestVector(TaskData *taskData, int *layers, int numOfLayers, float *values, float *expectedOutput) {
    if (taskData->testLine >= taskData->totalTestLines) {
        return false;
    }
    for (int input = 0; input < layers[0]; input++) {
        values[input] = taskData->testInputs[input + taskData->testLine * layers[0]];
    }

    for (int output = 0; output < layers[numOfLayers - 1]; output++) {
        expectedOutput[output] = taskData->testOutputs[output + taskData->testLine * layers[numOfLayers - 1]];
    }
    taskData->testLine ++;
    return true;
}

/**
 * Reads and stores input vectors for learning and testing neural network.
 */
void loadInputData(const char* learningFilename, const char* testFilename, int *layers, int numOfLayers, TaskData *taskData) {
    std::ifstream learningInput(learningFilename);

    float i1, i2, i3, i4, i5, i6, i7, i8, i9;
    int o1, o2;
    int totalLearningLines = std::count(std::istreambuf_iterator<char>(learningInput), std::istreambuf_iterator<char>(), '\n');
    taskData->totalLearningLines = totalLearningLines;
    taskData->learningLine = 0;
    learningInput.seekg(0);
    // std::cout << totalLearningLines << std::endl;
    taskData->learningInputs = (float *) malloc (totalLearningLines * layers[0] * sizeof(float));
    taskData->learningOutputs = (float *) malloc (totalLearningLines * layers[numOfLayers - 1] * sizeof(float));
    unsigned long int learningInputCounter = 0;
    unsigned long int learningOutputCounter = 0;
    while (learningInput >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> i8 >> i9 >> o1 >> o2)
    {
        taskData->learningInputs[learningInputCounter] = i1; learningInputCounter++;
        taskData->learningInputs[learningInputCounter] = i2; learningInputCounter++;
        taskData->learningInputs[learningInputCounter] = i3; learningInputCounter++;
        taskData->learningInputs[learningInputCounter] = i4; learningInputCounter++;
        taskData->learningInputs[learningInputCounter] = i5; learningInputCounter++;
        taskData->learningInputs[learningInputCounter] = i6; learningInputCounter++;
        taskData->learningInputs[learningInputCounter] = i7; learningInputCounter++;
        taskData->learningInputs[learningInputCounter] = i8; learningInputCounter++;
        taskData->learningInputs[learningInputCounter] = i9; learningInputCounter++;

        taskData->learningOutputs[learningOutputCounter] = o1; learningOutputCounter++;
        taskData->learningOutputs[learningOutputCounter] = o2; learningOutputCounter++;
    }

    std::ifstream testInput(learningFilename);
    int totalTestLines = std::count(std::istreambuf_iterator<char>(testInput), std::istreambuf_iterator<char>(), '\n');
    taskData->totalTestLines = totalTestLines;
    taskData->testLine = 0;
    testInput.seekg(0);
    // std::cout << totalTestLines << std::endl;
    taskData->testInputs = (float *) malloc (totalTestLines * layers[0] * sizeof(float));
    taskData->testOutputs = (float *) malloc (totalTestLines * layers[numOfLayers - 1] * sizeof(float));
    unsigned long int testInputCounter = 0;
    unsigned long int testOutputCounter = 0;
    while (testInput >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> i8 >> i9 >> o1 >> o2)
    {
        taskData->testInputs[testInputCounter] = i1; testInputCounter++;
        taskData->testInputs[testInputCounter] = i2; testInputCounter++;
        taskData->testInputs[testInputCounter] = i3; testInputCounter++;
        taskData->testInputs[testInputCounter] = i4; testInputCounter++;
        taskData->testInputs[testInputCounter] = i5; testInputCounter++;
        taskData->testInputs[testInputCounter] = i6; testInputCounter++;
        taskData->testInputs[testInputCounter] = i7; testInputCounter++;
        taskData->testInputs[testInputCounter] = i8; testInputCounter++;
        taskData->testInputs[testInputCounter] = i9; testInputCounter++;

        taskData->testOutputs[testOutputCounter] = o1; testOutputCounter++;
        taskData->testOutputs[testOutputCounter] = o2; testOutputCounter++;
    }
}

/**
 * Performs trainign and testing of neural network.
 */
void runNeuralNetwork(float learnFactor) {
    int neuronsPerLayer = 32;
    int numOfLayers = 4;
    int layers[] = {9, 29, 19, 2};
    // int layers[] = {9, 9, 9, 2};
    int *accurancy; 
    
    float *weights, *values, *errors, *expectedOutput;
    TaskData taskData;

    int numOfWeights = 0;
    for (int i = 1; i < numOfLayers; i++) {
        numOfWeights += layers[i] * layers[i - 1];
    }

    weights = (float *) malloc (numOfWeights * sizeof(float));
    initWeights(weights, numOfWeights);

    accurancy = (int *) malloc (4 * layers[numOfLayers - 1] * sizeof(int));
    initAccurancy(accurancy, 4 * layers[numOfLayers - 1]);
    
    int numOfValues = 0;
    for (int i = 0; i < numOfLayers; i++) {
        numOfValues += layers[i];
    }
    values =  (float *) malloc (numOfValues * sizeof(float));
    errors =  (float *) malloc (numOfValues * sizeof(float));
    
    expectedOutput =  (float *) malloc (layers[numOfLayers - 1] * sizeof(float));

    loadInputData("cancerL.dt", "cancerT.dt", layers, numOfLayers, &taskData);
    
    // learning
    while (getLearningVector(&taskData, layers, numOfLayers, values, expectedOutput)) {
        neuralLearnCycle(values, weights, expectedOutput, errors, layers, numOfLayers, learnFactor, 1);
        if (taskData.learningLine > 340 && taskData.learningLine < 350) {
            // printNeuralNetwork(values, weights, expectedOutput, errors, layers, numOfLayers);
        }
    }

    //testing
    while (getTestVector(&taskData, layers, numOfLayers, values, expectedOutput)) {
        neuralTestCycle(values, weights, expectedOutput, layers, numOfLayers, 1, accurancy);
        if (taskData.testLine > 340 && taskData.testLine < 350) {
            // printNeuralNetwork(values, weights, expectedOutput, errors, layers, numOfLayers);
        }
    }

    printAccurancy(accurancy, 4 * layers[numOfLayers - 1]);
    // std::cout << learnFactor <<" : " << accurancy[0]+accurancy[3]+accurancy[4]+accurancy[7] <<std::endl;
    free(weights);
    free(values);
    free(errors);
    free(expectedOutput);
    free(accurancy);
    free(taskData.learningInputs);
    free(taskData.learningOutputs);
    free(taskData.testInputs);
    free(taskData.testOutputs);
}


int main(int argc, char **argv)
{
    #if USE_PAPI
    papi_routines.AddRoutine("network_learning_cycle");
    papi_routines.AddRoutine("network_test_cycle");
    #endif

    // for (float l = 5.3; l < 5.5; l +=0.01) {
        // runNeuralNetwork(l);
        runNeuralNetwork(0.4);
    // }

    #if USE_PAPI
    papi_routines.PrintScreen();
    #endif
    return 0;
}