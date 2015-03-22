#include "initPapi.h"

#if USE_PAPI_LEARN_AND_TEST || USE_PAPI_LEARN_DETAIL || USE_PAPI_TEST_DETAIL || USE_PAPI_NEURAL_ROW_LEARN || USE_PAPI_NEURAL_ROW_TEST
PapiCounterList *papi_routines;
#endif

void initPapi() {
    #if USE_PAPI_LEARN_AND_TEST || USE_PAPI_LEARN_DETAIL || USE_PAPI_TEST_DETAIL || USE_PAPI_NEURAL_ROW_LEARN || USE_PAPI_NEURAL_ROW_TEST
    papi_routines = new PapiCounterList();
    #endif

    #if USE_PAPI_LEARN_AND_TEST
    papi_routines->AddRoutine("network_learning");
    papi_routines->AddRoutine("network_testing");
    papi_routines->AddRoutine("o_network_learning");
    papi_routines->AddRoutine("o_network_testing");
    #endif

    #if USE_PAPI_LEARN_DETAIL
    papi_routines->AddRoutine("network_learning_foward_computation");
    papi_routines->AddRoutine("network_learning_error_computation");
    papi_routines->AddRoutine("network_learning_weight_update");
    papi_routines->AddRoutine("o_network_learning_foward_computation");
    papi_routines->AddRoutine("o_network_learning_error_computation");
    papi_routines->AddRoutine("o_network_learning_weight_update");
    #endif

    #if USE_PAPI_TEST_DETAIL
    papi_routines->AddRoutine("network_testing_foward_computation");
    papi_routines->AddRoutine("network_testing_error_computation");
    papi_routines->AddRoutine("o_network_testing_foward_computation");
    papi_routines->AddRoutine("o_network_testing_error_computation");
    #endif 

    #if USE_PAPI_NEURAL_ROW_LEARN
    papi_routines->AddRoutine("network_learning_neural_row_foward_computation");
    papi_routines->AddRoutine("network_learning_neural_row_error_computation");
    papi_routines->AddRoutine("network_learning_neural_row_weight_update");
    papi_routines->AddRoutine("o_network_learning_neural_row_foward_computation");
    papi_routines->AddRoutine("o_network_learning_neural_row_error_computation");
    papi_routines->AddRoutine("o_network_learning_neural_row_weight_update");
    #endif

    #if USE_PAPI_NEURAL_ROW_TEST
    papi_routines->AddRoutine("network_testing_neural_row_foward_computation");
    papi_routines->AddRoutine("network_testing_neural_row_error_computation");
    papi_routines->AddRoutine("o_network_testing_neural_row_foward_computation");
    papi_routines->AddRoutine("o_network_testing_neural_row_error_computation");
    #endif  
}
