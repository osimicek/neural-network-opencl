#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <papi.h>
#include "papi_cntr.h"

#define USE_PAPI 1

struct TaskData {
   float *learningInputs, *learningOutputs;
   int totalLearningLines;
   float *testInputs, *testOutputs;
   int totalTestLines;
};

struct Setup {
    int numOfLayers;
    int *layers;
    bool classification;
    float minOutputValue;
    float maxOutputValue;
    float learningFactor;
    float lambda;
};

struct Criteria {
    int maxEpochs;
    float minProgress;
    float maxGeneralizationLoss;
};

struct State {
    int epoch;
    float *weights;
    float *values;
    float *errors;
    int testLine;
    int learningLine;
};

struct NeuralNetwork {
    Setup setup;
    Criteria criteria;
    State state;
    float bestAccurancy[2]; // [epochID, accurancy]
    float *accurancyHistory; // [accurancy1, accurancy2, ...].length = maxEpochs
    int *classificationAccurancyHistory;
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

void printNeuralNetwork(NeuralNetwork *neuralNetwork,
                        float *expectedOutput) {
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
void printAccurancy(int *accurancy, int numOfNeurons) {
    for (int neuron = 0; neuron < numOfNeurons; neuron++) {
        printf("N. %d\t| true T | true F | precision\n", neuron / 4);
        printf("pred. T\t| %d\t | %d\t  | %5.1f%%\n", accurancy[neuron * 4], accurancy[neuron * 4 + 1], accurancy[neuron * 4] / ((accurancy[neuron * 4] + accurancy[neuron * 4 + 1]) / 100.0));
        printf("pred. F\t| %d\t | %d\t  | %5.1f%%\n", accurancy[neuron * 4 + 2], accurancy[neuron * 4 + 3], accurancy[neuron * 4 + 3] / ((accurancy[neuron * 4 + 2] + accurancy[neuron * 4 + 3]) / 100.0));
        printf("recall\t| %5.1f%% | %5.1f%% \n\n", accurancy[neuron * 4] / ((accurancy[neuron * 4] + accurancy[neuron * 4 + 2]) / 100.0),
                                      accurancy[neuron * 4 + 3] / ((accurancy[neuron * 4 + 1] + accurancy[neuron * 4 + 3]) / 100.0));
    }
}


/**
 * Returns overall accuracy.
 */
float getTotalAccurancy(int *accurancy, int numOfNeurons) {
    float total = 0.f;
    int correct = 0;
    for (int neuron = 0; neuron < numOfNeurons; neuron++) {
        total += accurancy[neuron * 4] + accurancy[neuron * 4 + 1] + accurancy[neuron * 4 + 2] + accurancy[neuron * 4 + 3];
        correct += accurancy[neuron * 4] + accurancy[neuron * 4 + 3];
    }
    return correct/total;
}

/**
 * Performs a learning cycle of neural network. Input vector is transformed to output vector.
 * Output vector si compared with expected output and error is backpropagated throw net and
 * weights are modified.
 */
void neuralLearnCycle(NeuralNetwork *neuralNetwork, 
                float *expectedOutput) {
    #if USE_PAPI
    papi_routines["network_learning_cycle"].Start();
    #endif
    int numOfLayers = neuralNetwork->setup.numOfLayers;
    int *layers = neuralNetwork->setup.layers;
    float *values = neuralNetwork->state.values;
    float *weights = neuralNetwork->state.weights;
    float *errors = neuralNetwork->state.errors;


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
            values[neuron + valueOffset] = 1. / (1. + exp(-neuralNetwork->setup.lambda * value));
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
            // std::cout << value << " "; 

        if (expectedOutput[neuron] - boolValue) {
            // std::cout << "chyba u " << neuron << " " << value << std::endl;
            float ex = 1. / (1. + exp(-neuralNetwork->setup.lambda * value));;
            // std::cout << "velikost ch " << (expectedOutput[neuron] - value) * (ex * (1 - ex)) << " ex " << ex << std::endl;
            errors[neuron + valueOffset] = (expectedOutput[neuron] - value) * (ex * (1 - ex));
        } else {
            errors[neuron + valueOffset] = 0;
        }
        // std::cout << (expectedOutput[neuron] - value) << std::endl;
    }
    // std::cout  << std::endl;
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
            float ex = exp(-neuralNetwork->setup.lambda*value);
            errors[neuron + valueOffset] = weightError * (ex * (1 - ex));
        }
        followValueOffset -= neurons;
    }
    // printNeuralNetwork(neuralNetwork, expectedOutput);
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
                                                     neuralNetwork->setup.learningFactor * values[prevNeuron + valueOffset] *
                                                     errors[neuron + valueOffset + prevNeurons];
            }
            weightOffset += prevNeurons;
        }
        valueOffset += prevNeurons;
    }

    // printNeuralNetwork(neuralNetwork, expectedOutput);

    #if USE_PAPI
    papi_routines["network_learning_cycle"].Stop();
    #endif
}

/**
 * Performs a test cycle of neural network. Input vector is transformed to output vector.
 * Output vector si compared with expected output. Accurancy is counted from this error.
 */
void neuralTestCycle(NeuralNetwork *neuralNetwork, 
                float *expectedOutput,
                int *accurancy) {
    #if USE_PAPI
    papi_routines["network_test_cycle"].Start();
    #endif
    int numOfLayers = neuralNetwork->setup.numOfLayers;
    int *layers = neuralNetwork->setup.layers;
    float *values = neuralNetwork->state.values;
    float *weights = neuralNetwork->state.weights;

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
            values[neuron + valueOffset] = 1. / (1. + exp(-neuralNetwork->setup.lambda * value));
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
bool getLearningVector(NeuralNetwork *neuralNetwork, TaskData *taskData, float *expectedOutput) {

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
bool getTestVector(NeuralNetwork *neuralNetwork, TaskData *taskData, float *expectedOutput) {

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
 * Reads and stores input vectors for learning and testing neural network.
 */
void loadInputData(const char* learningFilename, const char* testFilename, NeuralNetwork *neuralNetwork, TaskData *taskData) {
    std::ifstream learningInput(learningFilename);

    float i1, i2, i3, i4, i5, i6, i7, i8, i9;
    int o1, o2;
    int totalLearningLines = std::count(std::istreambuf_iterator<char>(learningInput), std::istreambuf_iterator<char>(), '\n');
    taskData->totalLearningLines = totalLearningLines;
    neuralNetwork->state.learningLine = 0;
    learningInput.seekg(0);
    // std::cout << totalLearningLines << std::endl;
    taskData->learningInputs = (float *) malloc (totalLearningLines * neuralNetwork->setup.layers[0] * sizeof(float));
    taskData->learningOutputs = (float *) malloc (totalLearningLines * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * sizeof(float));
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
    neuralNetwork->state.testLine = 0;
    testInput.seekg(0);
    // std::cout << totalTestLines << std::endl;
    taskData->testInputs = (float *) malloc (totalTestLines * neuralNetwork->setup.layers[0] * sizeof(float));
    taskData->testOutputs = (float *) malloc (totalTestLines * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1] * sizeof(float));
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
void runNeuralNetwork(float learningFactor) {
    NeuralNetwork neuralNetwork;
    neuralNetwork.setup.lambda = 1;
    neuralNetwork.setup.learningFactor = learningFactor;
    neuralNetwork.setup.numOfLayers = 4;
    int tmpLayers[] = {9, 9, 9, 2};
    neuralNetwork.setup.layers = tmpLayers;

    neuralNetwork.criteria.maxEpochs = 25;
    neuralNetwork.criteria.minProgress = 0.0001f;
    neuralNetwork.criteria.maxGeneralizationLoss = 0.05;
    int *accurancy; 
    
    float *expectedOutput;
    TaskData taskData;

    int numOfWeights = 0;
    for (int i = 1; i < neuralNetwork.setup.numOfLayers; i++) {
        numOfWeights += neuralNetwork.setup.layers[i] * neuralNetwork.setup.layers[i - 1];
    }

    neuralNetwork.state.weights = (float *) malloc (numOfWeights * sizeof(float));
    initWeights(neuralNetwork.state.weights, numOfWeights);

    int accurancyLength = 4 * neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1];
    accurancy = (int *) malloc (accurancyLength * sizeof(int));
    initAccurancy(accurancy, 4 * neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1]);
    
    int numOfValues = 0;
    for (int i = 0; i < neuralNetwork.setup.numOfLayers; i++) {
        numOfValues += neuralNetwork.setup.layers[i];
    }
    neuralNetwork.state.values =  (float *) malloc (numOfValues * sizeof(float));
    neuralNetwork.state.errors =  (float *) malloc (numOfValues * sizeof(float));
    
    expectedOutput =  (float *) malloc (neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1] * sizeof(float));

    loadInputData("cancerL.dt", "cancerT.dt", &neuralNetwork, &taskData);
    float totalAccurancy = 0.f;
    float lastTotalAccurancy = -100.f;
    for (neuralNetwork.state.epoch = 0; neuralNetwork.state.epoch < neuralNetwork.criteria.maxEpochs; neuralNetwork.state.epoch++) {
        neuralNetwork.state.learningLine = 0;
        neuralNetwork.state.testLine = 0;
        // learning
        while (getLearningVector(&neuralNetwork, &taskData, expectedOutput)) {
            neuralLearnCycle(&neuralNetwork, expectedOutput);
            if (neuralNetwork.state.learningLine > 3 && neuralNetwork.state.learningLine < 350) {
                // printNeuralNetwork(neuralNetwork, expectedOutput);
                // return;
            }
        }
        initAccurancy(accurancy, accurancyLength);
        //testing
        while (getTestVector(&neuralNetwork, &taskData, expectedOutput)) {
            neuralTestCycle(&neuralNetwork, expectedOutput, accurancy);
            if (neuralNetwork.state.testLine > 340 && neuralNetwork.state.testLine < 350) {
                // printNeuralNetwork(neuralNetwork, expectedOutput);
            }
        }
        totalAccurancy = getTotalAccurancy(accurancy, neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1]);
        float epsilon = totalAccurancy - lastTotalAccurancy;
        // std::cout << "presnost " << totalAccurancy << " "<< epsilon<< std::endl; 
        if (epsilon < 0.0001f) {
            // break;
        }
        lastTotalAccurancy = totalAccurancy;
    }
    printf("%f  %f\n", neuralNetwork.setup.learningFactor, getTotalAccurancy(accurancy, neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1]));
    printAccurancy(accurancy, neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1]);

    // std::cout << learnFactor <<" : " << accurancy[0]+accurancy[3]+accurancy[4]+accurancy[7] <<std::endl;
    free(neuralNetwork.state.weights);
    free(neuralNetwork.state.values);
    free(neuralNetwork.state.errors);
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

    // for (float l = 0.01; l < 10; l +=0.1) {
        // runNeuralNetwork(l);
        runNeuralNetwork(0.4f);
    // }

    #if USE_PAPI
    papi_routines.PrintScreen();
    #endif
    return 0;
}