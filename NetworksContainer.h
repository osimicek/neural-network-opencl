#ifndef NETWORKS_CONTAINER_H
#define NETWORKS_CONTAINER_H
#include <vector>
#include "NeuralNetwork.h"

class NetworksContainer {
    public:
        std::vector<NeuralNetwork *> neural_networks_storage;
        std::vector<NeuralNetwork *> neural_networks;

        neural_network_transform_t *transforms;
        task_data_transform_t task_data_transform;
        int neural_network_transforms_size;
        void *neural_network_buffer;
        int neural_network_buffer_size;
        int shared_memory_per_network;
        taskData_t taskData;
        void *task_data_buffer;
        int task_data_buffer_size;
        int inputVectorSize;
        int outputVectorSize;
        int container_size;
        NetworksContainer();
        void alloc_neural_network_buffer();
        void init_networks();
        void load_input_data(const char* filename);
        void update_networks();
        void *get_neural_network_buffer();
        void *get_task_data_buffer();
        int get_neural_network_buffer_size();
        int get_task_data_buffer_size();
        int size();
        int get_number_of_neural_networks();
        task_data_transform_t *get_task_data_transform();
        neural_network_transform_t *get_transforms();
        int get_transforms_size();

};

#endif