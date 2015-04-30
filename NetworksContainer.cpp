#include <iostream>
#include <fstream> // file read
#include <mm_malloc.h>
#include "NetworksContainer.h"

#define MEMORY_ALIGN 128
/**
 * Networks container groups several neural networks together.
 * It allows common initialization of all networks and
 * creation of common buffer for OpenCL usage.
 */
NetworksContainer::NetworksContainer() {
    this->set_size(256);

    for (int i = 0; i < container_size; i++) {
        this->neural_networks_storage.push_back(new NeuralNetwork);
    }
    // this->neural_networks[0]->set_hidden_layers(1, 20);
    // this->neural_networks_storage[1]->set_hidden_layers(1, 20);
    this->neural_network_buffer = NULL;
    this->task_data_buffer = NULL;
    this->shared_memory_per_network = 4000;
}

/**
 * Prepares networks to run on GPU. Allocates common buffer
 */
void NetworksContainer::init_networks() {
    this->neural_networks.clear();

    // choose only networks that fit to shared memory
    for (std::vector<NeuralNetwork *>::iterator neural_network = this->neural_networks_storage.begin();
        neural_network != this->neural_networks_storage.end();
        neural_network++) {
        (*neural_network)->set_input_layer(this->inputVectorSize);
        (*neural_network)->set_output_layer(this->outputVectorSize);
        if ((*neural_network)->get_required_shared_memory_size() < this->shared_memory_per_network) {
            // network fits to shared memory
            this->neural_networks.push_back(*neural_network);
        }
    }


    this->neural_network_transforms_size = 0;
    int buffer_size = 0;
    for (std::vector<NeuralNetwork *>::iterator neural_network = this->neural_networks.begin();
        neural_network != this->neural_networks.end();
        neural_network++) {
        int index = neural_network - this->neural_networks.begin();
        this->transforms[index].neuralNetwork_b_offset = buffer_size;
        buffer_size += (*neural_network)->get_required_buffer_size();
    }

    if (this->neural_network_buffer != NULL) {
        free(this->neural_network_buffer);
    }
    this->neural_network_buffer_size = buffer_size * sizeof(float);
    this->neural_network_buffer = _mm_malloc(this->neural_network_buffer_size, MEMORY_ALIGN);
    if (this->neural_network_buffer == NULL) {
        fprintf(stderr,"Out of memory\n");
        exit(-1);
    }

    for (std::vector<NeuralNetwork *>::iterator neural_network = this->neural_networks.begin();
        neural_network != this->neural_networks.end();
        neural_network++) {
        int index = neural_network - this->neural_networks.begin();

        this->neural_network_transforms_size += sizeof(neural_network_transform_t);

        (*neural_network)->init(&this->transforms[index], (this->neural_network_buffer + this->transforms[index].neuralNetwork_b_offset * sizeof(float)));

        (*neural_network)->export_net(&this->transforms[index], (this->neural_network_buffer + this->transforms[index].neuralNetwork_b_offset * sizeof(float)), this->task_data_buffer, &this->taskData);
    }
}

/**
 * Update all networks witch transforms data.
 */
void NetworksContainer::update_networks() {
    for (std::vector<NeuralNetwork *>::iterator neural_network = this->neural_networks.begin();
        neural_network != this->neural_networks.end();
        neural_network++) {
        int index = neural_network - this->neural_networks.begin();
        (*neural_network)->import_net(&this->transforms[index], (this->neural_network_buffer + this->transforms[index].neuralNetwork_b_offset * sizeof(float)), this->task_data_buffer, &this->taskData);
    }
}

/**
 * Reads and stores input vectors for learning and testing neural network.
 */
void NetworksContainer::load_input_data(const char* filename) {
    std::cout << "ok--" << std::endl << std::flush;
    std::ifstream input(filename);
    
    uint inputVectorSize, outputVectorSize, totalLearningLines, totalTestLines;
    input >> inputVectorSize;
    input >> outputVectorSize;
    input >> totalLearningLines;
    input >> totalTestLines;

    this->inputVectorSize = inputVectorSize;
    this->outputVectorSize = outputVectorSize;

    
    float value;
    unsigned long int learningInputCounter = 0;
    unsigned long int learningOutputCounter = 0;
    unsigned long int testInputCounter = 0;
    unsigned long int testOutputCounter = 0;
    int learningInputSize = totalLearningLines * inputVectorSize;
    int learningOutputSize = totalLearningLines * outputVectorSize;
    int testInputSize = totalTestLines * inputVectorSize;
    int testOutputSize = totalTestLines * outputVectorSize;
    this->task_data_buffer_size = (learningInputSize +
                            learningOutputSize +
                            testInputSize +
                            testOutputSize
                            ) * sizeof(float);
    this->task_data_buffer = _mm_malloc(this->task_data_buffer_size, MEMORY_ALIGN);
    if (this->task_data_buffer == NULL) {
        fprintf(stderr,"Out of memory\n");
        exit(-1);
    }
    this->taskData.learningInputs = (float *) this->task_data_buffer;
    this->taskData.learningOutputs = &((float *)this->task_data_buffer)[learningInputSize];
    this->taskData.testInputs = &((float *)this->task_data_buffer)[learningInputSize + learningOutputSize];
    this->taskData.testOutputs = &((float *)this->task_data_buffer)[learningInputSize + learningOutputSize + testInputSize];

    this->task_data_transform.taskData_b_offset_learningInputs = 0;
    this->task_data_transform.taskData_b_offset_learningOutputs = learningInputSize;
    this->task_data_transform.taskData_b_offset_testInputs = learningInputSize + learningOutputSize;
    this->task_data_transform.taskData_b_offset_testOutputs = learningInputSize + learningOutputSize + testInputSize;
    this->task_data_transform.taskData_b_size = learningInputSize + learningOutputSize + testInputSize + testOutputSize;
    this->task_data_transform.taskData_totalLearningLines = totalLearningLines;
    this->task_data_transform.taskData_totalTestLines = totalTestLines;

    for (unsigned int row = 0; row < totalLearningLines; row++) {
        // break;
        for (unsigned int inputCol = 0; inputCol < inputVectorSize; inputCol++) {
            input >> value;
            this->taskData.learningInputs[learningInputCounter] = value;
            learningInputCounter++;
        }

        for (unsigned int outputCol = 0; outputCol < outputVectorSize; outputCol++) {
            input >> value;
            this->taskData.learningOutputs[learningOutputCounter] = value;
            learningOutputCounter++;
        }
    }
    for (unsigned int row = 0; row < totalTestLines; row++) {
        // break;
        for (unsigned int inputCol = 0; inputCol < inputVectorSize; inputCol++) {
            input >> value;
            this->taskData.testInputs[testInputCounter] = value;
            testInputCounter++;
        }

        for (unsigned int outputCol = 0; outputCol < outputVectorSize; outputCol++) {
            input >> value;
            this->taskData.testOutputs[testOutputCounter] = value;
            testOutputCounter++;
        }
    }
    this->taskData.totalLearningLines = totalLearningLines;
    this->taskData.totalTestLines = totalTestLines;
    std::cout << "total: " << taskData.totalLearningLines << " " << taskData.totalTestLines << std::endl; 
    input.close();
}

/**
 * Returns common neural network buffer.
 */
void *NetworksContainer::get_neural_network_buffer() {
    return this->neural_network_buffer;
}

/**
 * Returns size of common neural network buffer.
 */
int NetworksContainer::get_neural_network_buffer_size() {
    return this->neural_network_buffer_size;
}

/**
 * Returns task data buffer.
 */
void *NetworksContainer::get_task_data_buffer() {
    return this->task_data_buffer;
}

/**
 * Returns size of task data buffer.
 */
int NetworksContainer::get_task_data_buffer_size() {
    return this->task_data_buffer_size;
}

/**
 * Sets size of container;
 */
int NetworksContainer::set_size(int size) {
    this->container_size = size;
    this->transforms = (neural_network_transform_t *) _mm_malloc(container_size * sizeof(neural_network_transform_t), MEMORY_ALIGN);
}

/**
 * Returns total size of container;
 */
int NetworksContainer::size() {
    return this->container_size;
}

/**
 * Returns number of neural networks that can be run.
 */
int NetworksContainer::get_number_of_neural_networks() {
    return this->neural_networks.size();
}

/**
 * Returns transform structure for task data.
 */
task_data_transform_t *NetworksContainer::get_task_data_transform() {
    return &this->task_data_transform;
}

/**
 * Returns array of transform structures for neural networks.
 */
neural_network_transform_t *NetworksContainer::get_transforms() {
    return this->transforms;
}

/**
 * Returns size of array of transform structures for neural networks.
 */
int NetworksContainer::get_transforms_size() {
    return this->neural_network_transforms_size;
}

/**
 * Returns required size of shared memory.
 */
int NetworksContainer::get_shared_memory_per_network() {
    return this->shared_memory_per_network;
}
/**
 * Returns required size of shared memory.
 */
std::vector<NeuralNetwork *> * NetworksContainer::get_neural_networks_storage() {
    return &this->neural_networks_storage;
}
