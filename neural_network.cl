#define OPENCL_MEMORY_ALIGN 5 // 2**OPENCL_MEMORY_ALIGN * sizeof(float)

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

size_t get_num_of_groups() {
    return  get_num_groups(0) * get_num_groups(1) * get_num_groups(2);
}

size_t get_group_size() {
    return  get_local_size(0) * get_local_size(1) * get_local_size(2);
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

    event_t e2 = async_work_group_copy(expectedOutput,
                                    (taskData->testOutputs +
                                        neuralNetwork->state.testLine * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1]),
                                    neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1], 0);
    wait_group_events(1, &e2);
    neuralNetwork->state.testLine++;
    return true;
}

/**
 * Reads and stores input vectors for neural network classification.
 */
bool getPredictVector(neuralNetwork_t *neuralNetwork, taskData_t *taskData, __global float **expectedOutput) {

    if (neuralNetwork->state.learningLine >= taskData->totalLearningLines) {
        return false;
    }

    event_t e = async_work_group_copy(neuralNetwork->state.values,
                                    (taskData->learningInputs +
                                        neuralNetwork->state.learningLine * neuralNetwork->setup.layers[0]),
                                    neuralNetwork->setup.layers[0], 0);
    wait_group_events(1, &e);

    *expectedOutput = taskData->learningOutputs + neuralNetwork->state.learningLine * neuralNetwork->setup.layers[neuralNetwork->setup.numOfLayers - 1];

    neuralNetwork->state.learningLine++;
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
                       __local float *tmp,
                       int neuron) {
    int numOfLayers = neuralNetwork->setup.numOfLayers;
    __local int * layers = neuralNetwork->setup.layers;
    __local float * values = neuralNetwork->state.values;
    __global float * weights = neuralNetwork->state.weights;
    __local float * errors = neuralNetwork->state.errors;


    int valueOffset = 0;
    int prevValueOffset = 0;
    int weightOffset = 0;

    int layer;
    float value;
    int neurons;
    int prevNeurons;
    int prevNeuron;
    int neuronsRounded;

    float weightError;
    // foward value computation
    for (layer = 1; layer < numOfLayers; layer++) {
        prevNeurons = layers[layer - 1];
        neurons = layers[layer];
        valueOffset += prevNeurons;
        neuronsRounded = (((neurons - 1) >> OPENCL_MEMORY_ALIGN) + 1) << OPENCL_MEMORY_ALIGN;
        if (neuron < neurons) {
            value = 0;
            for (prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                value += values[prevNeuron + prevValueOffset] * weights[neuron + weightOffset];
                weightOffset += neuronsRounded;
                prefetch(&weights[weightOffset], neurons);
            }
            values[neuron + valueOffset] = 1.f / (1.f + exp(-value));
        } else {
            weightOffset += prevNeurons * neuronsRounded;
        }
        prevValueOffset += prevNeurons;
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    // backwards error computation
    neurons = layers[numOfLayers - 1];
    if (neuron < neurons) {
        value = values[neuron + valueOffset];
        errors[neuron + valueOffset] = (expectedOutput[neuron] - value) * (value * (1 - value));
    }
    int maxNeurons = get_group_size();
    barrier(CLK_LOCAL_MEM_FENCE);
    prevValueOffset = valueOffset;
    for (layer = numOfLayers - 1; layer > 1; layer--) {
        neurons = layers[layer];
        neuronsRounded = (((neurons - 1) >> OPENCL_MEMORY_ALIGN) + 1) << OPENCL_MEMORY_ALIGN;
        prevNeurons = layers[layer - 1];
        weightOffset -= prevNeurons * neuronsRounded;
        prevValueOffset -= prevNeurons;
        for (prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
            prefetch(&weights[weightOffset], neurons);
            tmp[neuron] = 0.f;
            if (neuron < neurons) { // neuron represents weight
                tmp[neuron] = errors[neuron + valueOffset] * weights[neuron + weightOffset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
           
            for (int subTree = maxNeurons >> 1; subTree >= 1; subTree >>= 1) {
                if (neuron < subTree) {
                    tmp[neuron] = tmp[neuron] + tmp[neuron | subTree];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (neuron == 0) {
                weightError = tmp[0];
                value = values[prevNeuron + prevValueOffset];
                errors[prevNeuron + prevValueOffset] = weightError * (value * (1 - value));
            }
            weightOffset += neuronsRounded;
        }
        weightOffset -= prevNeurons * neuronsRounded;
        valueOffset = prevValueOffset;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // error propagation
    weightOffset = 0;
    valueOffset = 0;
    for (layer = 1; layer < numOfLayers; layer++) {
        prevNeurons = layers[layer - 1];
        neurons = layers[layer];
        neuronsRounded = (((neurons - 1) >> OPENCL_MEMORY_ALIGN) + 1) << OPENCL_MEMORY_ALIGN;
        // TOTO valueOffset = valueOffset + prevNeurons
        // prevValueOffset
        if (neuron < neurons) {
            for (prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                weights[neuron + weightOffset] += neuralNetwork->setup.learningFactor * values[prevNeuron + valueOffset] *
                                                      errors[neuron + valueOffset + prevNeurons];
                weightOffset += neuronsRounded;
                prefetch(&weights[weightOffset], neurons);   
            }
        } else {
            weightOffset += neuronsRounded*prevNeurons;
        }
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
    int neuronsRounded;
    int classValue;

    // foward value computation
    for (layer = 1; layer < numOfLayers; layer++) {
        prevNeurons = layers[layer - 1];
        neurons = layers[layer];
        valueOffset += prevNeurons;
        neuronsRounded = (((neurons - 1) >> OPENCL_MEMORY_ALIGN) + 1) << OPENCL_MEMORY_ALIGN;
        // prefetch(&weights[weightOffset], prevNeuronsRounded);
        if (neuron < neurons) {
            value = 0;
            for (prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                value += values[prevNeuron + prevValueOffset] * weights[neuron + weightOffset];
                weightOffset += neuronsRounded;
            }
            values[neuron + valueOffset] = 1.f / (1.f + exp(-value));
        } else {
            weightOffset += prevNeurons * neuronsRounded;
        }
        prevValueOffset += prevNeurons;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // error of classification
    neurons = layers[numOfLayers - 1];
    float classificationCandidateValue = -1.f;
    int classificationCandidateIndex = 0;
    for (int neur = 0; neur < neurons; neur++) {
        value = values[neur + valueOffset];
        classValue = round(value);
        if (classValue != 0 && classificationCandidateValue < value) {
            classificationCandidateValue = value;
            classificationCandidateIndex = neur;
        }
    }
    if (expectedOutput[classificationCandidateIndex] != 1.f || classificationCandidateValue == -1.f) {
        neuralNetwork->currentSquareErrorCounter += 1.f;
    }
}

/**
 * Performs a classification cycle of neural network. Input vector is transformed to output vector.
 * Output vector si stored in suplided data array.
 */
void neuralPredictCycle(neuralNetwork_t *neuralNetwork, 
                        __global float *expectedOutput,
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
    int neuronsRounded;
    int classValue;

    // foward value computation
    for (layer = 1; layer < numOfLayers; layer++) {
        prevNeurons = layers[layer - 1];
        neurons = layers[layer];
        valueOffset += prevNeurons;
        neuronsRounded = (((neurons - 1) >> OPENCL_MEMORY_ALIGN) + 1) << OPENCL_MEMORY_ALIGN;
        // prefetch(&weights[weightOffset], prevNeuronsRounded);
        if (neuron < neurons) {
            value = 0;
            for (prevNeuron = 0; prevNeuron < prevNeurons; prevNeuron++) {
                value += values[prevNeuron + prevValueOffset] * weights[neuron + weightOffset];
                weightOffset += neuronsRounded;
            }
            values[neuron + valueOffset] = 1.f / (1.f + exp(-value));
        } else {
            weightOffset += prevNeurons * neuronsRounded;
        }
        prevValueOffset += prevNeurons;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // error of classification
    neurons = layers[numOfLayers - 1];
    // expectedOutput[0] = 25;
    // return;
    if (neuron < neurons) {
        expectedOutput[neuron] = 0.f;
    }

    float classificationCandidateValue = -1.f;
    int classificationCandidateIndex = 0;
    for (int neur = 0; neur < neurons; neur++) {
        value = values[neur + valueOffset];
        classValue = round(value);
        if (classValue != 0 && classificationCandidateValue < value) {
            classificationCandidateValue = value;
            classificationCandidateIndex = neur;
        }
    }
    expectedOutput[classificationCandidateIndex] = 1.f;
    // expectedOutput[0] = values[valueOffset];
    // expectedOutput[1] = values[valueOffset + 1];
}


void importDataStructures(__global neural_network_transform_t *neural_network_transform,
                          __global task_data_transform_t *task_data_transform,
                          __global float * neural_network_buffer,
                          __global float *task_data_buffer,
                          neuralNetwork_t *neuralNetwork,
                          taskData_t *taskData) {
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

    // event_t e2 = async_work_group_copy(neuralNetwork->state.values,
    //                                 (neural_network_buffer + neural_network_transform->state_b_offset_values),
    //                                 neural_network_transform->state_b_size_values, 0);
    // wait_group_events(1, &e2);

    // event_t e3 = async_work_group_copy(neuralNetwork->state.errors,
    //                                 (neural_network_buffer + neural_network_transform->state_b_offset_errors),
    //                                 neural_network_transform->state_b_size_errors, 0);
    // wait_group_events(1, &e3);
    neuralNetwork->currentSquareErrorCounter = neural_network_transform->neuralNetwork_currentSquareErrorCounter;
    neuralNetwork->bestSquareError[0] = neural_network_transform->neuralNetwork_bestSquareError[0];
    neuralNetwork->bestSquareError[1] = neural_network_transform->neuralNetwork_bestSquareError[1];
    neuralNetwork->squareErrorHistory = (neural_network_buffer + neural_network_transform->neuralNetwork_b_offset_squareErrorHistory);
}

void exportDataStructures(__global neural_network_transform_t *neural_network_transform,
                        __global float * neural_network_buffer,
                        neuralNetwork_t *neuralNetwork) {
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

uint get_interupt_limit(neuralNetwork_t *neuralNetwork) {
    uint number_of_weights = 0;
    for (int i = 1; i < neuralNetwork->setup.numOfLayers; i++) {
        number_of_weights += neuralNetwork->setup.layers[i] * neuralNetwork->setup.layers[i - 1];
    }
    return 9000 / ((number_of_weights / 2000) + 1) + 1;
}

__kernel void run_neural_network(
    __global neural_network_transform_t *neural_network_transform_arr,
    __global float *neural_network_buffer_arr,
    __global task_data_transform_t *task_data_transform,
    __global float *task_data_buffer,
    int number_of_networks
    )
{
    __global float *neural_network_buffer = neural_network_buffer_arr;
    int group_id = get_group_linear_id();
    uint cycle_counter = 0;

    if (group_id < number_of_networks) {
    // if (group_id != 0) {
        int lid = get_local_linear_id();
        taskData_t taskData;
        neuralNetwork_t neuralNetwork;
        __global neural_network_transform_t *neural_network_transform = &neural_network_transform_arr[group_id];
        neural_network_buffer = neural_network_buffer + neural_network_transform->neuralNetwork_b_offset;
        __local float *expectedOutput;
        __local float tmp[256];
        __local float sharedMemory[SHARED_MEMORY_SIZE];
        neuralNetwork.setup.layers = (__local int*)sharedMemory;
        neuralNetwork.state.values = sharedMemory + neural_network_transform->setup_numOfLayers;
        neuralNetwork.state.errors = sharedMemory + neural_network_transform->setup_numOfLayers + neural_network_transform->state_b_size_values;
        importDataStructures(neural_network_transform, task_data_transform, neural_network_buffer, task_data_buffer, &neuralNetwork, &taskData);
        expectedOutput = sharedMemory + neural_network_transform->setup_numOfLayers + neural_network_transform->state_b_size_values + neural_network_transform->state_b_size_errors;
        uint interupt_limit = get_interupt_limit(&neuralNetwork) * 3; // 1 training cycle = 3, 1 learning cycle = 1 
        for (; neuralNetwork.state.epoch < neuralNetwork.criteria.maxEpochs; neuralNetwork.state.epoch++) {
            while (getLearningVector(&neuralNetwork, &taskData, expectedOutput)) {
                neuralLearnCycle(&neuralNetwork, expectedOutput, tmp, lid);
                cycle_counter += 3;
                if (cycle_counter >= interupt_limit) {
                    exportDataStructures(neural_network_transform, neural_network_buffer, &neuralNetwork);
                    return;
                }
                // if (neuralNetwork.state.learningLine >= 15  && neuralNetwork.state.epoch >= 0) {
                //     exportDataStructures(neural_network_transform, neural_network_buffer, &neuralNetwork);
                //     return;
                // }
            }

            while (getTestVector(&neuralNetwork, &taskData, expectedOutput)) {
                neuralTestCycle(&neuralNetwork, expectedOutput, lid);
                cycle_counter++;
                if (cycle_counter >= interupt_limit) {
                    exportDataStructures(neural_network_transform, neural_network_buffer, &neuralNetwork);
                    return;
                }
                // if (neuralNetwork.state.testLine > 348  && neuralNetwork.state.epoch >= 0) {
                    // exportDataStructures(neural_network_transform, neural_network_buffer, &neuralNetwork);
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
        exportDataStructures(neural_network_transform, neural_network_buffer, &neuralNetwork);
    }
}



__kernel void run_neural_network_classification(
    __global neural_network_transform_t *neural_network_transform,
    __global float *neural_network_buffer,
    __global task_data_transform_t *task_data_transform,
    __global float *task_data_buffer,
    int number_of_networks
    )
{
    int group_id = get_group_linear_id();
    int total_groups = get_num_of_groups();
    uint cycle_counter = 0;
    if (group_id < number_of_networks) {
    // if (group_id != 0) {
        int lid = get_local_linear_id();
        taskData_t taskData;
        neuralNetwork_t neuralNetwork;
        __global float *expectedOutput;
        __local float sharedMemory[SHARED_MEMORY_SIZE];
        neuralNetwork.setup.layers = (__local int*)sharedMemory;
        neuralNetwork.state.values = sharedMemory + neural_network_transform->setup_numOfLayers;
        importDataStructures(neural_network_transform, task_data_transform, neural_network_buffer, task_data_buffer, &neuralNetwork, &taskData);
        int num_of_vectors = taskData.totalLearningLines;
        int batch_size = num_of_vectors / total_groups;
        neuralNetwork.state.learningLine = batch_size * group_id;
        if (group_id != total_groups - 1) {
            taskData.totalLearningLines = batch_size * (group_id + 1);
        }
        while (getPredictVector(&neuralNetwork, &taskData, &expectedOutput)) {
            neuralPredictCycle(&neuralNetwork, expectedOutput, lid);
            // return;
            cycle_counter++;
            // if (neuralNetwork.state.learningLine >= 15  && neuralNetwork.state.epoch >= 0) {
            //     exportDataStructures(neural_network_transform, neural_network_buffer, &neuralNetwork);
            //     return;
            // }
        }
    }
}
