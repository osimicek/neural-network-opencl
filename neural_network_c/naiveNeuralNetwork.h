#ifndef NAIVE_NEURAL_NETWORK_H
#define NAIVE_NEURAL_NETWORK_H

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
    float currentSquareErrorCounter;  // sum of square errors
    float bestSquareError[2]; // [epochID, accurancy]     avg square error
    float *squareErrorHistory; // [accurancy1, accurancy2, ...].length = maxEpochs    avg square error
    int *classificationAccurancyHistory;
};
namespace naive {
    void runNaiveNeuralNetwork(float learningFactor);
    void printClassificationAccurancy(int *accurancy, int numOfNeurons);
    void initAccurancy(int *accurancy, int length);
    void findAndSetBestSquareError(NeuralNetwork *neuralNetwork);
    float squareErrorSum(NeuralNetwork *neuralNetwork);

    bool getLearningVector(NeuralNetwork *neuralNetwork, TaskData *taskData, float *expectedOutput);
    bool getTestVector(NeuralNetwork *neuralNetwork, TaskData *taskData, float *expectedOutput);
    void loadInputData(const char* filename, NeuralNetwork *neuralNetwork, TaskData *taskData);
    void deleteTaskData(TaskData *taskData);
}
#endif
