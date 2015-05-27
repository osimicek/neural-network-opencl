#ifndef NETWORKS_RUNNER_H
#define NETWORKS_RUNNER_H
#include "OpenclHelper.h"
#include "NetworkContainer.h"
#include "NeuralNetwork.h"

class NetworkRunner {
    private:
        NetworkContainer *networks_container;
        Device *device;
        CommandQueue *queue;
        Context *ctx;
        Kernel *knl_learn;
        Kernel *knl_classification;
        Buffer *buf_task_data_transform;
        Buffer *buf_taskdata;
        int max_neurons;
        void *task_data_buffer;
        int task_data_buffer_size;
        bool has_all_finished(neural_network_transform_t * transforms, int number_of_networks);
    public:
        NetworkRunner(uint platformId, uint deviceId, NetworkContainer *networks_container);
        ~NetworkRunner();
        void set_max_neurons(int value);
        void write_task_data();
        void run_networks(bool verbose = false);
        void run_networks_classification(int number_of_networks, bool verbose = false);
};

#endif