#include "initPapi.h"
#include "naiveNeuralNetwork.h"

int main(int argc, char **argv)
{
    initPapi();
    
    // for (float l = 0.1; l < 10; l +=0.1) {
    //     runNeuralNetwork(l);
    // }
    runNaiveNeuralNetwork(0.4f);

    #if USE_PAPI_LEARN_AND_TEST || USE_PAPI_LEARN_DETAIL || USE_PAPI_TEST_DETAIL
    papi_routines->PrintScreen();
    delete papi_routines;
    #endif
    return 0;
}
