#include <iostream>
#include "OpenclHelper.h"
#include "NeuralNetwork.h"
#include "NetworksContainer.h"
#include "NetworksRunner.h"
#include "GeneticAlgorithm.h"

int main(int argc, char **argv)
{
    // OpenclHelper::print_platforms_devices(true);

    NetworksContainer networks_container;
    networks_container.load_input_data("./neural_network_c/data/cancer.dt");

    NetworksRunner networks_runner(&networks_container);
    networks_runner.write_task_data();
    // networks_runner.run_networks();
    GeneticAlgorithm ga(&networks_container, &networks_runner);
    ga.init();
    ga.run();

    // free(neural_network_buffer);
    // free(task_data_buffer);
} 