#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H


/**
 * Stores task data.
 */
typedef struct {
   float *learningInputs, *learningOutputs;
   int totalLearningLines;
   float *testInputs, *testOutputs;
   int totalTestLines;
} taskData_t;


/**
 * Neural network base structures.
 */
typedef struct {
    int numOfLayers;
    int *layers;
    bool classification;
    float minOutputValue;
    float maxOutputValue;
    float learningFactor;
    float lambda;
} setup_t;

typedef struct {
    int maxEpochs;
    float minProgress;
    float maxGeneralizationLoss;
} criteria_t;

typedef struct {
    int epoch;
    float *weights;
    float *values;
    float *errors;
    int testLine;
    int learningLine;
} state_t;

// typedef struct {
//     setup_t setup;
//     criteria_t criteria;
//     state_t state;
//     float currentSquareErrorCounter;
//     float bestSquareError[2];
//     float *squareErrorHistory;
// } neuralNetwork_t;

/**
 * Schema used to recreate taskData structure in kernel.
 */
typedef struct {
    int taskData_b_offset_learningInputs, taskData_b_offset_learningOutputs;
    int taskData_totalLearningLines;
    int taskData_b_offset_testInputs, taskData_b_offset_testOutputs;
    int taskData_totalTestLines;
    int taskData_b_size;
} task_data_transform_t;

/**
 * Schema used to recreate neuralNetwork structure in kernel.
 */
typedef struct {
    int setup_numOfLayers;
    int setup_b_offset_layers;
    bool setup_classification;
    float setup_minOutputValue;
    float setup_maxOutputValue;
    float setup_learningFactor;
    float setup_lambda;

    int criteria_maxEpochs;
    float criteria_minProgress;
    float criteria_maxGeneralizationLoss;

    int state_epoch;
    int state_b_offset_weights;
    int state_b_offset_values;
    int state_b_size_values;
    int state_b_offset_errors;
    int state_b_size_errors;
    int state_testLine;
    int state_learningLine;

    float neuralNetwork_currentSquareErrorCounter;  
    float neuralNetwork_bestSquareError[2];
    int neuralNetwork_b_offset_squareErrorHistory;
    int neuralNetwork_b_size;
    int neuralNetwork_b_offset;
} neural_network_transform_t;

class NeuralNetwork {
    private:
        bool allocatedLayers;
        void init_weights(int length);

    public:
        setup_t setup;
        criteria_t criteria;
        state_t state;
        float *squareErrorHistory;
        float bestSquareError[2];
        float currentSquareErrorCounter;
        NeuralNetwork();
        ~NeuralNetwork();
        void print(float *expectedOutput);
        void import_net(neural_network_transform_t *transform,
                        void * neural_network_buffer,
                        void *task_data_buffer,
                        taskData_t *taskData);
        void export_net(neural_network_transform_t *transform, 
                        void * neural_network_buffer,
                        void *task_data_buffer,
                        taskData_t *taskData);
        void init(neural_network_transform_t *transform, void *neural_network_buffer);
        void set_hidden_layers(int numberOfHiddenLayers, int numberOfNeuronsPerLayer);
        void set_max_epochs(int epochs);
        void set_input_layer(int neurons);
        void set_output_layer(int neurons);
        void set_learning_factor(float learning_factor);
        int get_required_buffer_size();
        int get_required_shared_memory_size();
        float get_best_square_error();
        int get_best_epoch();
};

#endif