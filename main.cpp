#include <iostream>
#include "OpenclHelper.h"
#include "NeuralNetwork.h"
#include "NetworksContainer.h"
#include "NetworksRunner.h"

int main(int argc, char **argv)
{
    NetworksContainer networks_container;
    networks_container.load_input_data("./neural_network_c/data/cancer.dt");

    OpenclHelper::print_platforms_devices(true);

    NetworksRunner networks_runner(&networks_container);
    networks_runner.write_task_data(&networks_container);
    networks_runner.run_networks(&networks_container);

    // delete netw;
    /**
     * Allocate mem
     */
    
    
    // free(neural_network_buffer);
    // free(task_data_buffer);
} 