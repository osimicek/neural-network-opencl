#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H
#include <vector>
#include "NetworksRunner.h"
#include "NetworksContainer.h"

class GeneticAlgorithm {
    private:
        float best_measure;
        int best_number_of_neurons;
        int best_number_of_hidden_layers;
        int number_of_elite;
        float best_learning_factor;
    public:
        NetworksContainer *networks_container;
        NetworksRunner *networks_runner;
        std::vector<uint> chromosomes;
        std::vector<float> measures;
        std::vector<float> fitnesses;
        int population_size;
        int generation;
        int max_generations;
        float reproduction_probability;
        float mutation_probability;

        GeneticAlgorithm(NetworksContainer *container, NetworksRunner *runner);
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
        void run();
};

#endif
