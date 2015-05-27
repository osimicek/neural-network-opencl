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

struct NeuralNetworkT {
    Setup setup;
    Criteria criteria;
    State state;
    float currentSquareErrorCounter;  // sum of square errors
    float bestSquareError[2]; // [epochID, accurancy]     avg square error
    float *squareErrorHistory; // [accurancy1, accurancy2, ...].length = maxEpochs    avg square error
    int *classificationAccurancyHistory;
};
namespace naive {
    void runNaiveNeuralNetwork(NeuralNetworkT *neuralNetwork, const char* taskFilename);
    void printClassificationAccurancy(int *accurancy, int numOfNeurons);
    void initAccurancy(int *accurancy, int length);
    void findAndSetBestSquareError(NeuralNetworkT *neuralNetwork);
    float squareErrorSum(NeuralNetworkT *neuralNetwork);

    bool getLearningVector(NeuralNetworkT *neuralNetwork, TaskData *taskData, float *expectedOutput);
    bool getTestVector(NeuralNetworkT *neuralNetwork, TaskData *taskData, float *expectedOutput);
    bool getClassificationVector(NeuralNetworkT *neuralNetwork, TaskData *taskData, float **classificationOutput);
    void loadInputData(const char* filename, NeuralNetworkT *neuralNetwork, TaskData *taskData);
    void loadClassificationData(const char* filename, NeuralNetworkT *neuralNetwork, TaskData *taskData);
    void storeClassification(const char* filename, NeuralNetworkT *neuralNetwork, TaskData *taskData);
    void deleteTaskData(TaskData *taskData);
}
#endif
