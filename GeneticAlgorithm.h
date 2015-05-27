#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H
#include <vector>
#include "NetworkRunner.h"
#include "NetworkContainer.h"

class GeneticAlgorithm {
    private:
        float best_measure;
        int number_of_elite;
    public:
        int best_number_of_neurons;
        int best_number_of_hidden_layers;
        int best_number_of_epochs;
        float best_learning_factor;
        NetworkContainer *networks_container;
        NetworkRunner *networks_runner;
        std::vector<uint> chromosomes;
        std::vector<float> measures;
        std::vector<float> fitnesses;
        int population_size;
        int generation;
        int max_generations;
        int min_layers;
        int max_layers;
        int max_neurons;
        int network_epochs;
        float reproduction_probability;
        float mutation_probability;

        GeneticAlgorithm(NetworkContainer *container, NetworkRunner *runner);
        void init();
        float get_learning_factor(uint chromosome);
        int get_number_of_neurons(uint chromosome);
        int get_number_of_hidden_layers(uint chromosome);
        uint gray_to_binary(uint value);
        void crossover(uint *chromosomeA, uint *chromosomeB);
        void mutation(uint *chromosome);
        bool choose_chromosome(std::vector<uint*> *evolution_chromosomes,
                               std::vector<float> *evolution_fitnesses,
                               uint **chromosome);
        void generate_fitnesses();
        void evolution();
        void run(bool verbose = false);
        void set_max_generations(int value);
        void set_min_layers(int value);
        void set_max_layers(int value);
        void set_max_neurons(int value);
        void set_population_size(int value);
        void set_network_epochs(int value);
};

#endif
