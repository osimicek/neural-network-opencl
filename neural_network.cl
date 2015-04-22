typedef struct {
   __global float *learningInputs, *learningOutputs;
   int totalLearningLines;
   __global float *testInputs, *testOutputs;
   int totalTestLines;
} taskData_t;

typedef struct {
    int numOfLayers;
    __local int *layers;
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
    __global float *weights;
    __local float *values;
    __local float *errors;
    int testLine;
    int learningLine;
} state_t;

typedef struct {
    setup_t setup;
    criteria_t criteria;
    state_t state;
    float currentSquareErrorCounter;
    float bestSquareError[2];
    __global float *squareErrorHistory;
} neuralNetwork_t;

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

typedef struct {
    int taskData_b_offset_learningInputs, taskData_b_offset_learningOutputs;
    int taskData_totalLearningLines;
    int taskData_b_offset_testInputs, taskData_b_offset_testOutputs;
    int taskData_totalTestLines;
    int taskData_b_size;
} task_data_transform_t;

size_t get_local_linear_id() {
    return  (get_local_id(2) * get_local_size(1) *
            (0)) + (get_local_id(1) *
            get_local_size(0)) + get_local_id(0);
    }

size_t get_group_linear_id() {
    return  (get_group_id(2) * get_num_groups(1) *
            (0)) + (get_group_id(1) *
            get_num_groups(0)) + get_group_id(0);
    }


/**
     * Returns a learning vector for neural network.
     */
bool getLearningVector(neuralNetwork_t *neuralNetwork, taskData_t *taskData, __local float *expectedOutput) {

    if (neuralNetwork->state.learningLine >= taskData->totalLearningLines) {
        return false;
    }

    event_t e = async_work_group_copy(neuralNetwork->state.values,
                                    (taskData->learningInputs +
                                        neuralNetwork->state.learningLine * neuralNetwork->setup.layers[0]),
                                    neuralNetwork->setup.layers[0], 0);
    wait_group_events(1, &e);
    for (int input = 0; input < neuralNetwork->setup.layers[0]; input++) {
        neuralNetwork->state.values[input] = taskData->learningInputs[input + neuralNetwork->state.learningLine * neuralNetwork->setup.layers[0]];
    }

    event_t e2 = async_work_group_copy(expectedOutput,
                                    (taskData->learningOutputs +
                                        neuralNetwork->state.learningLine * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]),
                                    neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1], 0);
    wait_group_events(1, &e2);

    neuralNetwork->state.learningLine++;
    return true;
}

/**
     * Returns a test vector for neural network.
     */
bool getTestVector(neuralNetwork_t *neuralNetwork, taskData_t *taskData, __local float *expectedOutput) {

    if (neuralNetwork->state.testLine >= taskData->totalTestLines) {
        return false;
    }
    event_t e = async_work_group_copy(neuralNetwork->state.values,
                                    (taskData->testInputs +
                                        neuralNetwork->state.testLine * neuralNetwork->setup.layers[0]),
                                    neuralNetwork->setup.layers[0], 0);
    wait_group_events(1, &e);
    for (int input = 0; input < neuralNetwork->setup.layers[0]; input++) {
        neuralNetwork->state.values[input] = taskData->testInputs[input + neuralNetwork->state.testLine * neuralNetwork->setup.layers[0]];
    }

    event_t e2 = async_work_group_copy(expectedOutput,
                                    (taskData->testOutputs +
                                        neuralNetwork->state.testLine * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]),
                                    neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1], 0);
    wait_group_events(1, &e2);
    neuralNetwork->state.testLine++;
    return true;
}

/**
 * Finds lowest square error and sets it as the best one.
 */
void findAndSetBestSquareError(neuralNetwork_t *neuralNetwork) {
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
 * Performs a learning cycle of neural network. Input vector is transformed to output vector.
 * Output vector si compared with expected output and error is backpropagated throw net and
 * weights are modified..
 */
void neuralLearnCycle(neuralNetwork_t *neuralNetwork, 
                       __local float *expectedOutput,
                       int neuron) {
    int numOfLayers = neuralNetwork->setup.numOfLayers;
    __local int * layers = neuralNetwork->setup.layers;
    __local float * values = neuralNetwork->state.values;
    __global float * weights = neuralNetwork->state.weights;
    __local float * errors = neuralNetwork->state.errors;


    int valueOffset = 0;
    int prevValueOffset = 0;
    int followValueOffset = 0;
    int weightOffset = 0;

    int layer;
    float value;
    int neurons;
    int prevNeurons;
    int prevNeuron;
    int followNeurons;
    int followNeuron;

    float ex;
    int classValue;
    float weightError;


    // foward value computation
    for (layer = 1; layer < numOfLayers; layer++) {
        prevNeurons = layers[layer - 1];
        prefetch(&weights[weightOffset], prevNeurons);
        neurons = layers[layer];
        valueOffset += prevNeurons;
        weightOffset += neuron * prevNeurons;
        if (neuron < neurons) {
            value = 0;
            for (prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                value += values[prevNeuron + prevValueOffset] * weights[prevNeuron + weightOffset];
            }
            values[neuron + valueOffset] = 1. / (1. + exp(-neuralNetwork->setup.lambda * value));
        }
        weightOffset += (neurons - neuron) * prevNeurons;
        prevValueOffset += prevNeurons;
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    // backwards error computation
    neurons = layers[numOfLayers - 1];
    if (neuron < neurons) {
        value = values[neuron + valueOffset];
        // classValue = value;
        // if (neuralNetwork->setup.classification) {
        //     classValue = round(value);
        // }

        // if (expectedOutput[neuron] - classValue) {
            float ex = 1.f / (1.f + exp(-neuralNetwork->setup.lambda * value));
            errors[neuron + valueOffset] = (expectedOutput[neuron] - value) * (ex * (1 - ex));
        // } else {
        //     errors[neuron + valueOffset] = 0;
        // }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    followValueOffset = valueOffset;
    for (layer = numOfLayers - 2; layer > 0; layer--) {
        followNeurons = layers[layer + 1];
        prefetch(&weights[weightOffset], followNeurons);
        neurons = layers[layer];
        valueOffset -= neurons;
        weightOffset -= (neurons - neuron) * followNeurons;
        if (neuron < neurons) {
            weightError = 0.f;
            for (followNeuron = 0; followNeuron < followNeurons; followNeuron++) {
                weightError += errors[followNeuron + followValueOffset] *
                              weights[followNeuron + weightOffset];
            }
            value = values[neuron + valueOffset];
            ex = exp(-neuralNetwork->setup.lambda * value);
            errors[neuron + valueOffset] = weightError * (ex * (1 - ex));
        }
        weightOffset -= neuron * followNeurons;
        followValueOffset -= neurons;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // error propagation
    weightOffset = 0;
    valueOffset = 0;
    for (layer = 1; layer < numOfLayers; layer++) {
        prevNeurons = layers[layer - 1];
        prefetch(&weights[weightOffset], prevNeurons);
        neurons = layers[layer];
        weightOffset += neuron * prevNeurons;
        if (neuron < neurons) {
            for (prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                weights[prevNeuron + weightOffset] = weights[prevNeuron + weightOffset] +
                                                     neuralNetwork->setup.learningFactor * values[prevNeuron + valueOffset] *
                                                     errors[neuron + valueOffset + prevNeurons];
            }
        }
        weightOffset += (neurons - neuron) * prevNeurons;
        valueOffset += prevNeurons;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/**
 * Performs a test cycle of neural network. Input vector is transformed to output vector.
 * Output vector si compared with expected output. Accurancy is counted from this error.
 */
void neuralTestCycle(neuralNetwork_t *neuralNetwork, 
                     __local float *expectedOutput,
                     int neuron) {
    
    int numOfLayers = neuralNetwork->setup.numOfLayers;
    __local int *layers = neuralNetwork->setup.layers;
    __local float *values = neuralNetwork->state.values;
    __global float *weights = neuralNetwork->state.weights;

    int valueOffset = 0;
    int prevValueOffset = 0;
    int weightOffset = 0;

    int layer;
    float value;
    int neurons;
    int prevNeurons;
    int prevNeuron;

    int classValue;
    float diff; 

    // foward value computation
    for (layer = 1; layer < numOfLayers; layer++) {
        neurons = layers[layer];
        prevNeurons = layers[layer - 1];
        valueOffset += prevNeurons;
        weightOffset += neuron * prevNeurons;
        if (neuron < neurons) {
            value = 0.f;
            for (prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                value += values[prevNeuron + prevValueOffset] * weights[prevNeuron + weightOffset];
            }
            values[neuron + valueOffset] = 1.f / (1.f + exp(-neuralNetwork->setup.lambda * value));
        }
        weightOffset += (neurons - neuron) * prevNeurons;
        prevValueOffset += prevNeurons;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // error of prediction
    neurons = layers[numOfLayers - 1];
    float predictionCandidateValue = -1.f;
    int predictionCandidateIndex = 0;
    for (int neur = 0; neur < neurons; neur++) {
        value = values[neur + valueOffset];
        classValue = round(value);
        if (classValue != 0 && predictionCandidateValue < value) {
            predictionCandidateValue = value;
            predictionCandidateIndex = neur;
        }
    }
    if (expectedOutput[predictionCandidateIndex] != 1.f || predictionCandidateValue == -1.f) {
        neuralNetwork->currentSquareErrorCounter += 1.f;
    }
}

void importDataStructures(__global neural_network_transform_t *neural_network_transform,
                          __global task_data_transform_t *task_data_transform,
                          __global float * neural_network_buffer,
                          __global float *task_data_buffer,
                          neuralNetwork_t *neuralNetwork,
                          taskData_t *taskData) {
    int lid = get_local_linear_id();
    taskData->learningInputs = (task_data_buffer + task_data_transform->taskData_b_offset_learningInputs);
    taskData->learningOutputs = (task_data_buffer + task_data_transform->taskData_b_offset_learningOutputs);
    taskData->totalLearningLines = task_data_transform->taskData_totalLearningLines;
    taskData->testInputs = (task_data_buffer + task_data_transform->taskData_b_offset_testInputs);
    taskData->testOutputs = (task_data_buffer + task_data_transform->taskData_b_offset_testOutputs);
    taskData->totalTestLines = task_data_transform->taskData_totalTestLines;

    neuralNetwork->setup.numOfLayers = neural_network_transform->setup_numOfLayers;
    neuralNetwork->setup.classification = neural_network_transform->setup_classification;
    neuralNetwork->setup.minOutputValue = neural_network_transform->setup_minOutputValue;
    neuralNetwork->setup.maxOutputValue = neural_network_transform->setup_maxOutputValue;
    neuralNetwork->setup.learningFactor = neural_network_transform->setup_learningFactor;
    neuralNetwork->setup.lambda = neural_network_transform->setup_lambda;
    event_t e1 = async_work_group_copy(neuralNetwork->setup.layers,
                                    (__global int *)(neural_network_buffer + neural_network_transform->setup_b_offset_layers),
                                    neural_network_transform->setup_numOfLayers, 0);
    wait_group_events(1, &e1);

    neuralNetwork->criteria.maxEpochs = neural_network_transform->criteria_maxEpochs;
    neuralNetwork->criteria.minProgress = neural_network_transform->criteria_minProgress;
    neuralNetwork->criteria.maxGeneralizationLoss = neural_network_transform->criteria_maxGeneralizationLoss;

    neuralNetwork->state.epoch = neural_network_transform->state_epoch;
    neuralNetwork->state.testLine = neural_network_transform->state_testLine;
    neuralNetwork->state.learningLine = neural_network_transform->state_learningLine;
    neuralNetwork->state.weights = (neural_network_buffer + neural_network_transform->state_b_offset_weights);
    // return;
    event_t e2 = async_work_group_copy(neuralNetwork->state.values,
                                    (neural_network_buffer + neural_network_transform->state_b_offset_values),
                                    neural_network_transform->state_b_size_values, 0);
    wait_group_events(1, &e2);

    event_t e3 = async_work_group_copy(neuralNetwork->state.errors,
                                    (neural_network_buffer + neural_network_transform->state_b_offset_errors),
                                    neural_network_transform->state_b_size_errors, 0);
    wait_group_events(1, &e3);
    neuralNetwork->currentSquareErrorCounter = neural_network_transform->neuralNetwork_currentSquareErrorCounter;
    neuralNetwork->bestSquareError[0] = neural_network_transform->neuralNetwork_bestSquareError[0];
    neuralNetwork->bestSquareError[1] = neural_network_transform->neuralNetwork_bestSquareError[1];
    neuralNetwork->squareErrorHistory = (neural_network_buffer + neural_network_transform->neuralNetwork_b_offset_squareErrorHistory);
}

void exportDataStructures(__global neural_network_transform_t *neural_network_transform,
                        __global float * neural_network_buffer,
                        __global float *task_data_buffer,
                        neuralNetwork_t *neuralNetwork,
                        taskData_t *taskData) {
    int lid = get_local_linear_id();
    neural_network_transform->setup_numOfLayers = neuralNetwork->setup.numOfLayers;
    neural_network_transform->setup_classification = neuralNetwork->setup.classification;
    neural_network_transform->setup_minOutputValue = neuralNetwork->setup.minOutputValue;
    neural_network_transform->setup_maxOutputValue = neuralNetwork->setup.maxOutputValue;
    neural_network_transform->setup_learningFactor = neuralNetwork->setup.learningFactor;
    neuralNetwork->setup.lambda = neural_network_transform->setup_lambda;
    event_t e1 = async_work_group_copy((__global int *)(neural_network_buffer + neural_network_transform->setup_b_offset_layers),
                                        neuralNetwork->setup.layers,
                                        neural_network_transform->setup_numOfLayers, 0);
    wait_group_events(1, &e1);

    neural_network_transform->criteria_maxEpochs = neuralNetwork->criteria.maxEpochs;
    neural_network_transform->criteria_minProgress = neuralNetwork->criteria.minProgress;
    neural_network_transform->criteria_maxGeneralizationLoss = neuralNetwork->criteria.maxGeneralizationLoss;

    neural_network_transform->state_epoch = neuralNetwork->state.epoch;
    neural_network_transform->state_testLine = neuralNetwork->state.testLine;
    neural_network_transform->state_learningLine = neuralNetwork->state.learningLine;
    event_t e2 = async_work_group_copy((neural_network_buffer + neural_network_transform->state_b_offset_values),
                                        neuralNetwork->state.values,
                                        neural_network_transform->state_b_size_values, 0);
    wait_group_events(1, &e2);

    event_t e3 = async_work_group_copy((neural_network_buffer + neural_network_transform->state_b_offset_errors),
                                        neuralNetwork->state.errors,
                                        neural_network_transform->state_b_size_errors, 0);
    wait_group_events(1, &e3);
    neural_network_transform->neuralNetwork_currentSquareErrorCounter = neuralNetwork->currentSquareErrorCounter;

    neural_network_transform->neuralNetwork_bestSquareError[0] = neuralNetwork->bestSquareError[0];
    neural_network_transform->neuralNetwork_bestSquareError[1] = neuralNetwork->bestSquareError[1];
}

__kernel void run_neural_network(
    __global neural_network_transform_t *neural_network_transform_arr,
    __global task_data_transform_t *task_data_transform,
    __global float *neural_network_buffer_arr,
    __global float *task_data_buffer,
    int number_of_networks
    )
{
    __global float *neural_network_buffer = neural_network_buffer_arr;
    int group_id = get_group_linear_id();
    int cycle_counter = 0;

    if (group_id < number_of_networks) {
    // if (group_id != 0) {
        int lid = get_local_linear_id();
        taskData_t taskData;
        neuralNetwork_t neuralNetwork;
        __global neural_network_transform_t *neural_network_transform = &neural_network_transform_arr[group_id];
        neural_network_buffer = neural_network_buffer + neural_network_transform->neuralNetwork_b_offset;
        __local float *expectedOutput;
        __local float sharedMemory[SHARED_MEMORY_SIZE];
        neuralNetwork.setup.layers = (__local int*)sharedMemory;
        neuralNetwork.state.values = sharedMemory + neural_network_transform->setup_numOfLayers;
        neuralNetwork.state.errors = sharedMemory + neural_network_transform->setup_numOfLayers + neural_network_transform->state_b_size_values;

        importDataStructures(neural_network_transform, task_data_transform, neural_network_buffer, task_data_buffer, &neuralNetwork, &taskData);
        expectedOutput = sharedMemory + neural_network_transform->setup_numOfLayers + neural_network_transform->state_b_size_values + neural_network_transform->state_b_size_errors;
        for (; neuralNetwork.state.epoch < neuralNetwork.criteria.maxEpochs; neuralNetwork.state.epoch++) {
            while (getLearningVector(&neuralNetwork, &taskData, expectedOutput)) {
                neuralLearnCycle(&neuralNetwork, expectedOutput, lid);
                cycle_counter++;
                if (cycle_counter > 10000) {
                    exportDataStructures(neural_network_transform, neural_network_buffer, task_data_buffer, &neuralNetwork, &taskData);
                    return;
                }
                // if (neuralNetwork.state.learningLine >= 0  && neuralNetwork.state.epoch >= 0) {
                //     exportDataStructures(neural_network_transform, neural_network_buffer, task_data_buffer, &neuralNetwork, &taskData);
                //     return;
                // }
            }
            // if (lid == 0){
            // }


            while (getTestVector(&neuralNetwork, &taskData, expectedOutput)) {
                neuralTestCycle(&neuralNetwork, expectedOutput, lid);
                cycle_counter++;
                if (cycle_counter > 10000) {
                    exportDataStructures(neural_network_transform, neural_network_buffer, task_data_buffer, &neuralNetwork, &taskData);
                    return;
                }
                // if (neuralNetwork.state.testLine > 348  && neuralNetwork.state.epoch >= 0) {
                    // exportDataStructures(neural_network_transform, neural_network_buffer, task_data_buffer, &neuralNetwork, &taskData);
                    // return;
                // }
            }
            neuralNetwork.squareErrorHistory[neuralNetwork.state.epoch] = (neuralNetwork.setup.maxOutputValue - neuralNetwork.setup.minOutputValue) *
                                                                            neuralNetwork.currentSquareErrorCounter / (neuralNetwork.setup.layers[neuralNetwork.setup.numOfLayers - 1] *
                                                                            taskData.totalTestLines);
            findAndSetBestSquareError(&neuralNetwork);
            neuralNetwork.currentSquareErrorCounter = 0.f;
            neuralNetwork.state.learningLine = 0;
            neuralNetwork.state.testLine = 0;
            // break;
        }
        exportDataStructures(neural_network_transform, neural_network_buffer, task_data_buffer, &neuralNetwork, &taskData);
    }
}
