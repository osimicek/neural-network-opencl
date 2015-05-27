#include <iostream>
#include <algorithm>
#include <math.h>
#include "GeneticAlgorithm.h"

/**
 * Genetic algorihm for neural networks.
 * Chromosomes are 32b integers. First 16b represents learnig factor. It is floating point value
 * stored as 3b.13b. Minimum value is 0 and maximum is 8.8192, step is 0.00012. Number of neurons
 * is represented with 10b and number of layers with 6b. All value in chromosome is represented
 * in Gray code.
 *
 * Chromosome:
 *  |                               32b                              |
 *  |      16b (learning factor)        10b (neurons)    6b (layers) |
 *  |--------------------------------|------------------|------------|
 */
GeneticAlgorithm::GeneticAlgorithm( NetworkContainer *container,
                                    NetworkRunner *runner):networks_container(container),networks_runner(runner) {
    this->generation = 0;
    this->set_population_size(256);

    this->max_generations = 1;
    this->reproduction_probability = 0.9;
    this->mutation_probability = 0.05;
    this->number_of_elite = 2; // elitism
    this->network_epochs = 10;
    this->min_layers = 0;
    this->max_layers = 4;
    this->max_neurons = 256;
}

void GeneticAlgorithm::init() {
    this->chromosomes.clear();
    this->measures.clear();
    this->fitnesses.clear();
    this->best_measure = 1;
    // srand(37);
    for (int i = 0; i < this->population_size; i++) {
        this->chromosomes.push_back(rand());
        this->measures.push_back(1);
        this->fitnesses.push_back(0.01);
    }
}

/**
 * Converts chromosome value coded in gray code to binary
 */
uint GeneticAlgorithm::gray_to_binary(uint value) {
    unsigned int mask;
    for (mask = value >> 1; mask != 0; mask = mask >> 1) {
        value = value ^ mask;
    }
    return value;
}

/**
 * Returns learning factor coded in chromosome
 */
float GeneticAlgorithm::get_learning_factor(uint chromosome) {
    uint value = (chromosome & 0xffff0000) >> 16;
    value = this->gray_to_binary(value);
    return value / 8192.;
}

/**
 * Returns number of neurons coded in chromosome
 */
int GeneticAlgorithm::get_number_of_neurons(uint chromosome) {
    uint value = (chromosome & 0x0000ffc0) >> 6;
    value = value >> 2; // lower 10b to 8b
    value = this->gray_to_binary(value);
    return value % (this->max_neurons + 1);
}

/**
 * Returns number of hidden layers coded in chromosome
 */
int GeneticAlgorithm::get_number_of_hidden_layers(uint chromosome) {
    uint value = chromosome & 0x0000003f;
    value = value >> 4; // lower 6b to 2b
    value = this->gray_to_binary(value);
    if (value < this->min_layers) {
        value = this->min_layers;
    }
    return value % (this->max_layers + 1);
}

/**
 * Performs corrsover of 2 chromosomes
 */
void GeneticAlgorithm::crossover(uint *chromosomeA, uint *chromosomeB) {
    int bitA, bitB;
    // learning factor
    uint r = rand() % 17;
    for (int i = 31 - r; i >= 16; i--) {
        bitA = ((*chromosomeA) >> i) & 1;
        bitB = ((*chromosomeB) >> i) & 1;
        *chromosomeA ^= (-bitB ^ (*chromosomeA)) & (1 << i);
        *chromosomeB ^= (-bitA ^ (*chromosomeB)) & (1 << i);
    }

    // number of neurons
    r = rand() % 11;
    for (int i = 15 - r; i >= 6; i--) {
        bitA = ((*chromosomeA) >> i) & 1;
        bitB = ((*chromosomeB) >> i) & 1;
        *chromosomeA ^= (-bitB ^ (*chromosomeA)) & (1 << i);
        *chromosomeB ^= (-bitA ^ (*chromosomeB)) & (1 << i);
    }

    // number of hidden networks
    r = rand() % 7;
    for (int i = 5 - r; i >= 0; i--) {
        bitA = ((*chromosomeA) >> i) & 1;
        bitB = ((*chromosomeB) >> i) & 1;
        *chromosomeA ^= (-bitB ^ (*chromosomeA)) & (1 << i);
        *chromosomeB ^= (-bitA ^ (*chromosomeB)) & (1 << i);
    }
}

/**
 * Performs mutation of chromosome
 */
void GeneticAlgorithm::mutation(uint *chromosome) {
    for (int i = 0; i < 32; i++) {
        float r = ((double) rand() / (RAND_MAX));
        if (r < this->mutation_probability) {
            *chromosome ^= (1 << i);
        }
    }
}

/**
 * Quasi random choose of chromosome. Probability of choose depends on fintness value of chromosome.
 */
bool GeneticAlgorithm::choose_chromosome(std::vector<uint*> *evolution_chromosomes,
                                          std::vector<float> *evolution_fitnesses,
                                          uint **chromosome) {
    int controll_sum = 0;
    for(std::vector<float>::iterator it = evolution_fitnesses->begin(); it != evolution_fitnesses->end(); ++it) {
        controll_sum += *it;
    }
    if (controll_sum == -1 * (int)evolution_fitnesses->size()) {
        return false;
    }
    float fitnesses_sum = 0;
    float r;
    int index = 0;
    for(std::vector<float>::iterator it = evolution_fitnesses->begin(); it != evolution_fitnesses->end(); ++it) {
        if (*it >= 0) {
            fitnesses_sum += *it;
        }
    }
    if (fitnesses_sum > 0) {
        r = fmod(rand(), fitnesses_sum);
    } else {
        r = 0;
    }
    fitnesses_sum = 0;
    for(std::vector<float>::iterator it = evolution_fitnesses->begin(); it != evolution_fitnesses->end(); ++it) {
        if (*it >= 0) {
            fitnesses_sum += *it;
            if (r <= fitnesses_sum) {
                index = it - evolution_fitnesses->begin();
                break;
            }
        }
    }
    *chromosome = (*evolution_chromosomes)[index];
    (*evolution_fitnesses)[index] = -1.f;
    return true;
}

/**
 * Generates fitness values from measure values;
 */
void GeneticAlgorithm::generate_fitnesses() {
    float min_measure = 1, max_measure = 0;
    for(std::vector<float>::iterator measure = this->measures.begin();
        measure != this->measures.end();
        ++measure) {
        if ((*measure) >= 0) {
            if (min_measure > (*measure)) {
                min_measure = (*measure);
            }

            if (max_measure < (*measure)) {
                max_measure = (*measure);
            }
        }
    }

    float tmp_min_measure = min_measure;
    for(std::vector<float>::iterator measure = this->measures.begin();
        measure != this->measures.end();
        ++measure) {
        if ((*measure) < 0) {
            (*measure) = min_measure / 2;
            tmp_min_measure = min_measure / 2;
        }
    }
    min_measure = tmp_min_measure;
    min_measure *= 0.99;
    this->fitnesses.clear();
    for(std::vector<float>::iterator measure = this->measures.begin();
        measure != this->measures.end();
        ++measure) {
        float fitness = 1 / (min_measure - max_measure) *(0.99 * (*measure) + min_measure * 0.01 - max_measure);
        this->fitnesses.push_back(fitness);
    }
}

/**
 * Process evolution of all chromosomes. It consists of crossovers and mutations.
 */
void GeneticAlgorithm::evolution() {
    uint *chromosomeA, *chromosomeB;
    std::vector<uint*> evolution_chromosomes;
    std::vector<float> evolution_fitnesses;
    std::vector <uint>::iterator it = this->chromosomes.begin();
    for (uint i = 0; i < this->chromosomes.size(); i++) {
        evolution_fitnesses.push_back(this->fitnesses[i]);
        evolution_chromosomes.push_back(&(*it));
        ++it;
    }

    for (int elite_pair = 0; elite_pair < this->number_of_elite; elite_pair += 2) {
        uint best_index =  std::max_element(evolution_fitnesses.begin(), evolution_fitnesses.end()) - evolution_fitnesses.begin();
        // printf("Max: %d\n", best_index);
        evolution_fitnesses.erase(evolution_fitnesses.begin() + best_index);
        evolution_chromosomes.erase(evolution_chromosomes.begin() + best_index);
        best_index =  std::max_element(evolution_fitnesses.begin(), evolution_fitnesses.end()) - evolution_fitnesses.begin();
        // printf("Max: %d\n", best_index);
        evolution_fitnesses.erase(evolution_fitnesses.begin() + best_index);
        evolution_chromosomes.erase(evolution_chromosomes.begin() + best_index);
    }

    while(this->choose_chromosome(&evolution_chromosomes, &evolution_fitnesses, &chromosomeA) &&
          this->choose_chromosome(&evolution_chromosomes, &evolution_fitnesses, &chromosomeB)) {
        float r = ((double) rand() / (RAND_MAX));
        if (r < this->reproduction_probability) {
            this->crossover(chromosomeA, chromosomeB);
            this->mutation(chromosomeA);
            this->mutation(chromosomeB);
        }
    }
}

/**
 * Runs genetic algorithm with neural networks throw all generations.
 */
void GeneticAlgorithm::run(bool verbose) {
    std::vector<NeuralNetwork *> * neural_networks = this->networks_container->get_neural_networks_storage();
    this->init();
    if (verbose) {
        printf("STARTING GENETIC ALGORITHM\n");
    }
    for (generation = 0; generation < this->max_generations; generation++) {
        for(std::vector<NeuralNetwork *>::iterator neural_network = neural_networks->begin();
            neural_network != neural_networks->end();
            ++neural_network) {
            delete *neural_network;
        }
        neural_networks->clear();

        for(std::vector<uint>::iterator chromosome = this->chromosomes.begin();
            chromosome != this->chromosomes.end();
            ++chromosome) {
            NeuralNetwork *nn = new NeuralNetwork();
        // std::cout << "nn " << this->get_number_of_hidden_layers(*chromosome) << " " << this->get_number_of_neurons(*chromosome) << " "<< this->get_learning_factor(*chromosome) <<" for "<< *chromosome<< std::endl;

            // if (chromosome - this->chromosomes.begin() == 1) {
            //     nn->set_learning_factor(2.2f);

            // }
            nn->set_hidden_layers(this->get_number_of_hidden_layers(*chromosome), this->get_number_of_neurons(*chromosome));
            nn->set_learning_factor(this->get_learning_factor(*chromosome));
            nn->set_max_epochs(this->network_epochs);

            neural_networks->push_back(nn);
        }
        // printf("Best configuration: \n  square error: %f\n  learning factor: %f\n  neurons: %d\n  hidden layers: %d\n",
        //     this->best_measure ,this->best_learning_factor,this->best_number_of_neurons, this->best_number_of_hidden_layers);
        // std::cout << generation<<std::flush << std::endl;

        this->networks_runner->set_max_neurons(this->max_neurons);
        this->networks_runner->run_networks();

        // return;
        this->measures.clear();
        int index = 0;
        for(std::vector<NeuralNetwork *>::iterator neural_network = neural_networks->begin();
            neural_network != neural_networks->end();
            ++neural_network) {
            float measure = (*neural_network)->get_best_square_error();
            this->measures.push_back(measure);
            if (this->best_measure > measure) {
                uint chromosome = this->chromosomes[index];
                this->best_measure = measure;
                this->best_number_of_epochs = (*neural_network)->get_best_epoch();;
                this->best_number_of_neurons = this->get_number_of_neurons(chromosome);
                this->best_number_of_hidden_layers = this->get_number_of_hidden_layers(chromosome);
                this->best_learning_factor = this->get_learning_factor(chromosome);
            }
            index++;
        }
        if (verbose) {
            printf("%d%%\n", ((generation + 1) * 100) / this->max_generations);
        }

        this->generate_fitnesses();

        this->evolution();
    }
    if (verbose) {
        printf("100%%\n");
        printf("Best configuration: \n  square error: %f\n  learning factor: %f\n  neurons: %d\n  hidden layers: %d\n\n",
                this->best_measure ,this->best_learning_factor,this->best_number_of_neurons, this->best_number_of_hidden_layers);
    }
}

void GeneticAlgorithm::set_max_generations(int value) {
    this->max_generations = value;
}

void GeneticAlgorithm::set_min_layers(int value) {
    this->min_layers = value;
}

void GeneticAlgorithm::set_max_layers(int value) {
    this->max_layers = value;
}

void GeneticAlgorithm::set_max_neurons(int value) {
    this->max_neurons = value;
}

void GeneticAlgorithm::set_population_size(int value) {
    networks_container->set_size(value);
    this->population_size = value;
}

void GeneticAlgorithm::set_network_epochs(int value) {
    this->network_epochs = value;
}
