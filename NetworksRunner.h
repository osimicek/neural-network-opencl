#ifndef NETWORKS_RUNNER_H
#define NETWORKS_RUNNER_H
#include "OpenclHelper.h"
#include "NetworksContainer.h"
#include "NeuralNetwork.h"

class NetworksRunner {
    public:
        Device *device;
        CommandQueue *queue;
        Context *ctx;
        Kernel *knl;

        Buffer *buf_taskdata;
        void *task_data_buffer;
        int task_data_buffer_size;
        NetworksRunner(NetworksContainer *networks_container);
        ~NetworksRunner();
        void write_task_data(NetworksContainer *networks_container);
        bool has_all_finished(neural_network_transform_t * transforms, int number_of_networks);
        void run_networks(NetworksContainer *networks_container);
};

#endif