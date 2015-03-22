# C implementation of neural network

The purpose of this code is to analyze computing requirements of neural network and optimize the code.
PROBEN1 and MNIST task data were used for testing.

## Requirements
- PAPI library

## Usage
Makefile usage

### Compile
- `make`  - g++ without PAPI
- `make papi_overall`     - g++ with learning and testing PAPI counters
- `make papi_detail`      - g++ with detailed learning and testing PAPI counters
- `make papi_neural_row`  - g++ with learning and testing of first neural row PAPI counters

### Run
- `make run`  - NOPAPI | PAPI FP counters
- `make runm` - NOPAPI | PAPI memory counters