# THE GPU BASED ACCELERATION OF NEURAL NETWORKS
OpenCL implementation of neural network. Performs learning and classification process.
This project is for research purpouses only.

This implementation is based on learning several neural networks at once. A genetic algorithm is used to find     
appropriate neural network settings.

## Requirements
- OpenCL headers
- freeGlut

## Usage
After compilation with make there can be used these parameters.
./main
- -h: prints this help
- -i: prints available OpenCL platforms and devices
- -p: specify OpenCL platform
- -d: specify OpenCL device
- -g: number of generation of Genetic algorithm
- -w: number of parallel networks (size of population)
- -e: epochs of training
- -m: minimum layers
- -x: maximum layers
- -n: (maximum) number of neurons
- -t: path to training data set
- -c: path to classification data set
- -o: path to classification output
Bnchmark options -pdenwtco and:  
- -b: allow benchmark
- -l: number of layers