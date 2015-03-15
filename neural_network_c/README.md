# C implementation of neural network

The purpose of this code is to analyze computing requirements of neural network and optimize the code.
PROBEN1 data were used for testing.

## Requirements
- PAPI library

## Usage
Makefile usage

### Compile
- `make`  - g++ without PAPI
- `make papi1`  - g++ with PAPI learning and testing counters
- `make papi2`  - g++ with PAPI detail learning and testing counters

### Run
- `make run`  - NOPAPI | PAPI FP counters
- `make runm` - NOPAPI | PAPI memory counters