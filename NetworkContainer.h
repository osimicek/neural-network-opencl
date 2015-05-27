#ifndef NETWORKS_CONTAINER_H
#define NETWORKS_CONTAINER_H
#include <vector>
#include "NeuralNetwork.h"

class NetworkContainer {
    private:
        std::vector<NeuralNetwork *> neural_networks_storage;

        neural_network_transform_t *transforms;
        task_data_transform_t task_data_transform;
        int neural_network_transforms_size;
        void *neural_network_buffer;
        int neural_network_buffer_size;
        void *task_data_buffer;
        int task_data_buffer_size;
        int container_size;
        void alloc_neural_network_buffer();

    public:
        int shared_memory_per_network;
        int inputVectorSize;
        int outputVectorSize;
        std::vector<NeuralNetwork *> neural_networks;
        taskData_t taskData;
        NetworkContainer(int size = 256);
        ~NetworkContainer();
        void init_networks();
        void load_input_data(const char* filename);
        void load_classification_data(const char* filename);
        void store_classification(const char* filename);
        void update_networks();
        void *get_neural_network_buffer();
        void *get_task_data_buffer();
        int get_neural_network_buffer_size();
        int get_task_data_buffer_size();
        int set_size(int size);
        int size();
        int get_number_of_neural_networks();
        task_data_transform_t *get_task_data_transform();
        neural_network_transform_t *get_transforms();
        int get_transforms_size();
        int get_shared_memory_per_network();
        std::vector<NeuralNetwork *> *get_neural_networks_storage();
};

#endif