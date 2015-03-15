#ifndef PAPI_INIT_H
#define PAPI_INIT_H

#if USE_PAPI_LEARN_AND_TEST || USE_PAPI_LEARN_DETAIL || USE_PAPI_TEST_DETAIL
#include <papi.h>
#include "papi_cntr.h"

extern PapiCounterList *papi_routines;
#endif

void initPapi();

#endif
