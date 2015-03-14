#ifndef PAPI_INIT_H
#define PAPI_INIT_H
#include <papi.h>
#include "papi_cntr.h"

// class PapiCounterList;

extern PapiCounterList *papi_routines;

void initPapi();

#endif
