#ifndef NAIVE_NEURAL_NETWORK_H
#define NAIVE_NEURAL_NETWORK_H

struct TaskData;
struct Setup;
struct Criteria;
struct State;
struct NeuralNetwork;

void runNaiveNeuralNetwork(float learningFactor);
void printClassificationAccurancy(int *accurancy, int numOfNeurons);
void findAndSetBestSquareError(NeuralNetwork *neuralNetwork);
float squareErrorSum(NeuralNetwork *neuralNetwork);

bool getLearningVector(NeuralNetwork *neuralNetwork, TaskData *taskData, float *expectedOutput);
bool getTestVector(NeuralNetwork *neuralNetwork, TaskData *taskData, float *expectedOutput);
void loadInputData(const char* learningFilename, const char* testFilename, NeuralNetwork *neuralNetwork, TaskData *taskData);
void deleteNeuralNetwork(NeuralNetwork *neuralNetwork);
void deleteTaskData(TaskData *taskData);
#endif
