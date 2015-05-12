#ifndef NETWORKS_RUNNER_H
#define NETWORKS_RUNNER_H
#include "OpenclHelper.h"
#include "NetworksContainer.h"
#include "NeuralNetwork.h"

class NetworksRunner {
    private:
        NetworksContainer *networks_container;
        Device *device;
        CommandQueue *queue;
        Context *ctx;
        Kernel *knl_learn;
        Kernel *knl_predict;
        Buffer *buf_task_data_transform;
        Buffer *buf_taskdata;
        int max_neurons;
        void *task_data_buffer;
        int task_data_buffer_size;
        bool has_all_finished(neural_network_transform_t * transforms, int number_of_networks);
    public:
        NetworksRunner(uint platformId, uint deviceId, NetworksContainer *networks_container);
        ~NetworksRunner();
        void set_max_neurons(int value);
        void write_task_data();
        void run_networks(bool verbose = false);
        void run_networks_prediction(int number_of_networks, bool verbose = false);
};

#endif