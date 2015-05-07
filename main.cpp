#include <iostream>
#include <unistd.h>
#include "OpenclHelper.h"
#include "NeuralNetwork.h"
#include "NetworksContainer.h"
#include "NetworksRunner.h"
#include "GeneticAlgorithm.h"

void print_help() {
    printf ("The GPU Based Acceleration of Neural Networks.\n"
            "Usage:\n"
            "\t-h: prints this help\n"
            "\t-i: prints available OpenCL platforms and devices\n"
            "\t-p: specify OpenCL platform\n"
            "\t-d: specify OpenCL device\n"
            "\t-e: epochs of training\n"
            "\t-m: minimum layers\n"
            "\t-x: maximum layers\n"
            "\t-n: (maximum) number of neurons\n"
            "\t-w: number of parallel networks (size of population)\n"
            "Benchmark options -pdenw and:  \n"
            "\t-b: allow benchmark\n"
            "\t-l: number of layers\n"
            );
}

int main(int argc, char **argv) {
    const char *taks_path = "./neural_network_c/data/cancer.dt";
    char c;
    uint platform = 0;
    uint device = 0;
    uint max_layers = 4;
    uint min_layers = 0;
    uint epochs = 10;
    uint generations = 10;
    uint num_of_networks = 256;

    bool bench = false;
    uint neurons = 256;
    uint num_of_layers = 3;

    while ((c = getopt (argc, argv, "p:d:im:x:e:bn:l:w:t:g:h")) != -1) {
        switch (c) {
            case 'h':
                print_help();
                return 0;
            case 'p':
                platform = stoi(optarg);
                break;
            case 'd':
                device = stoi(optarg);
                break;
            case 'i':
                OpenclHelper::print_platforms_devices(true);
                printf ("\nSumarization:\n");
                OpenclHelper::print_platforms_devices(false);
                return 0;
            case 'm':
                min_layers = stoi(optarg);
                break;
            case 'x':
                max_layers = stoi(optarg);
                break;
            case 'e':
                epochs = stoi(optarg);
                break;
            case 'g':
                generations = stoi(optarg);
                break;
            case 'w':
                num_of_networks = stoi(optarg);
                break;
            case 'b':
                bench = true;
                break;
            case 'n':
                neurons = stoi(optarg);
                break;
            case 'l':
                num_of_layers = stoi(optarg);
                break;
            case 't':
                taks_path = optarg;
                break;
            case '?':
                fprintf (stderr, "Invalid argument.\n");
                return 1;
            default:
                abort ();
        }
    }
    NetworksContainer networks_container;
    networks_container.load_input_data(taks_path);
    NetworksRunner networks_runner(platform, device, &networks_container);
    networks_runner.write_task_data();
    if (bench) {
        std::vector<NeuralNetwork *> * neural_networks = networks_container.get_neural_networks_storage();
        networks_container.set_size(num_of_networks);
        for (int i = 0; i < num_of_networks; i++) {
            NeuralNetwork *nn = new NeuralNetwork;
            nn->set_hidden_layers(num_of_layers, neurons);
            nn->set_max_epochs(epochs);
            neural_networks->push_back(nn);
        }
        networks_runner.run_networks();
    } else {
        GeneticAlgorithm ga(&networks_container, &networks_runner);
        ga.set_max_generations(generations);
        ga.set_min_layers(min_layers);
        ga.set_max_layers(max_layers);
        ga.set_max_neurons(neurons);
        ga.set_population_size(num_of_networks);
        ga.set_network_epochs(epochs);
        ga.set_population_size(num_of_networks);
        ga.init();
        ga.run();
    }


    // free(neural_network_buffer);
    // free(task_data_buffer);
} 