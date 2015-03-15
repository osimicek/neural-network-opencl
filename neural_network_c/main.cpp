#include "initPapi.h"
#include "naiveNeuralNetwork.h"
#include "optimizedNeuralNetwork.h"

int main(int argc, char **argv)
{   
    #if USE_PAPI_LEARN_AND_TEST || USE_PAPI_LEARN_DETAIL || USE_PAPI_TEST_DETAIL
    initPapi();
    #endif
    
    // for (float l = 0.1; l < 10; l +=0.1) {
    //     runNeuralNetwork(l);
    // }
    naive::runNaiveNeuralNetwork(0.4f);
    optimized::runOptimizedNeuralNetwork(0.4f);

    #if USE_PAPI_LEARN_AND_TEST || USE_PAPI_LEARN_DETAIL || USE_PAPI_TEST_DETAIL
    papi_routines->PrintScreen();
    delete papi_routines;
    #endif
    return 0;
}
