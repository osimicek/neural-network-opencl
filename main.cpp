#include <iostream>
#include <unistd.h>
#include <time.h>
#include "OpenclHelper.h"
#include "NeuralNetwork.h"
#include "NetworkContainer.h"
#include "NetworkRunner.h"
#include "GeneticAlgorithm.h"

void print_help() {
    printf ("The GPU Based Acceleration of Neural Networks.\n"
            "Usage:\n"
            "\t-h: prints this help\n"
            "\t-i: prints available OpenCL platforms and devices\n"
            "\t-p: specify OpenCL platform\n"
            "\t-d: specify OpenCL device\n"
            "\t-g: number of generation of Genetic algorithm\n"
            "\t-w: number of parallel networks (size of population)\n"
            "\t-e: epochs of training\n"
            "\t-m: minimum layers\n"
            "\t-x: maximum layers\n"
            "\t-n: (maximum) number of neurons\n"
            "\t-t: path to training data set\n"
            "\t-c: path to classification data set\n"
            "\t-o: path to classification output\n"
            "Benchmark options -pdenw and:  \n"
            "\t-b: allow benchmark\n"
            "\t-l: number of layers\n"
            );
}

int main(int argc, char **argv) {
    srand (time(NULL));
    const char *taks_path = "./data/cancer.dt";
    const char *classification_path = "./data/cancer_classification.dt";
    const char *classification_output = "out.txt";
    char c;
    uint platform = 0;
    uint device = 0;
#ifdef DEVICE_ID
    device = DEVICE_ID;
#endif
#ifdef PLATFORM_ID
    platform = PLATFORM_ID;
#endif 
    uint max_layers = 4;
    uint min_layers = 0;
    uint epochs = 10;
    uint generations = 10;
    uint num_of_networks = 256;

    bool bench = false;
    uint neurons = 256;
    uint num_of_layers = 3;

    while ((c = getopt (argc, argv, "p:d:im:x:e:bn:l:w:t:g:c:o:h")) != -1) {
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
            case 'c':
                classification_path = optarg;
                break;
            case 'o':
                classification_output = optarg;
                break;
            case '?':
                fprintf (stderr, "Invalid argument.\n");
                return 1;
            default:
                abort ();
        }
    }
    NetworkContainer networks_container;
    networks_container.load_input_data(taks_path);
    
    if (bench) {
        int required_shared_memory = num_of_layers + 2 +
                                2 * (num_of_layers * neurons + networks_container.inputVectorSize + networks_container.outputVectorSize) + 
                                networks_container.outputVectorSize + epochs; // layers + errors + values + output + epochs
        networks_container.shared_memory_per_network = required_shared_memory;
        NetworkRunner networks_runner(platform, device, &networks_container);
        networks_runner.set_max_neurons(neurons);
        networks_runner.write_task_data();
        printf("Bench test: \n network: %dx%d,  num:%d,  epochs: %d\n", num_of_layers, neurons, num_of_networks, epochs);
        std::vector<NeuralNetwork *> * neural_networks = networks_container.get_neural_networks_storage();
        networks_container.set_size(num_of_networks);
        for (uint i = 0; i < num_of_networks; i++) {
            NeuralNetwork *nn = new NeuralNetwork;
            nn->set_hidden_layers(num_of_layers, neurons);
            nn->set_max_epochs(epochs);
            neural_networks->push_back(nn);
        }
        networks_runner.run_networks(true);

        // CLASSIFICATION
        int best_epoch = (*neural_networks)[0]->get_best_square_error();
        int best_learning_factor = (*neural_networks)[0]->setup.learningFactor;
        for (uint i = 0; i < num_of_networks; i++) {
            delete (*neural_networks)[i];
        }
        neural_networks->clear();
        networks_container.set_size(1);
        NeuralNetwork *nn = new NeuralNetwork;
        nn->set_hidden_layers(num_of_layers, neurons);
        nn->set_max_epochs(best_epoch + 1);
        nn->set_learning_factor(best_learning_factor);
        neural_networks->push_back(nn);
        networks_runner.run_networks();
        networks_container.load_classification_data(classification_path);
        networks_runner.write_task_data();
        networks_runner.run_networks_classification(num_of_networks, true);
        networks_container.store_classification(classification_output);

        delete nn;
    } else {
        NetworkRunner networks_runner(platform, device, &networks_container);
        networks_runner.write_task_data();
        GeneticAlgorithm ga(&networks_container, &networks_runner);
        ga.set_max_generations(generations);
        ga.set_min_layers(min_layers);
        ga.set_max_layers(max_layers);
        ga.set_max_neurons(neurons);
        ga.set_population_size(num_of_networks);
        ga.set_network_epochs(epochs);
        ga.set_population_size(num_of_networks);
        ga.init();
        ga.run(true);

        // CLASSIFICATION
        std::vector<NeuralNetwork *> * neural_networks = networks_container.get_neural_networks_storage();
        
        neural_networks->clear();
        networks_container.set_size(1);
        NeuralNetwork *nn = new NeuralNetwork;
        nn->set_hidden_layers(ga.best_number_of_hidden_layers, ga.best_number_of_neurons);
        nn->set_max_epochs(ga.best_number_of_epochs + 1);
        nn->set_learning_factor(ga.best_learning_factor);
        neural_networks->push_back(nn);
        networks_runner.run_networks();
        networks_container.load_classification_data(classification_path);
        networks_runner.write_task_data();
        networks_runner.run_networks_classification(num_of_networks, true);
        networks_container.store_classification(classification_output);

        delete nn;
    }


    // free(neural_network_buffer);
    // free(task_data_buffer);
} 