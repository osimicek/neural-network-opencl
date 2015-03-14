/**
Copyright (c) 2012, Swiss National Supercomputing Center (CSCS)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the Swiss National Supercomputing Center (CSCS) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "papi_cntr.h"

#include <papi.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>


std::string derivedStatName(DerivedStatistics statIDX){
    switch(statIDX){
        case Derived_FLIPS:
            return std::string("derived_FLIPS");
            break;
        case Derived_FLOPS:
            return std::string("derived_FLOPS");
            break;
        case Derived_DP_vector_FLOPS:
            return std::string("derived_DP_vector_FLOPS");
            break;
        case Derived_SP_vector_FLOPS:
            return std::string("derived_SP_vector_FLOPS");
            break;
        case Derived_L1_DMR:
            return std::string("derived_L1_DMR");
            break;
        case Derived_L2_DMR:
            return std::string("derived_L2_DMR");
            break;
        case Derived_L1_TMR:
            return std::string("derived_L1_TMR");
            break;
        case Derived_L2_TMR:
            return std::string("derived_L2_TMR");
            break;
        case Derived_L3_TMR:
            return std::string("derived_L3_TMR");
            break;    
        case Derived_Mem_Bandwidth:
            return std::string("Derived_Mem_Bandwidth");
            break;        
        case Derived_BANDWIDTH_SS:
            return std::string("derived_BANDWIDTH_SS");
            break;
        case Derived_BANDWIDTH_DS:
            return std::string("derived_BANDWIDTH_DS");
            break;
    }
    return std::string("");
}

int findString(std::vector<std::string> const& strVec, std::string str){
    std::vector<std::string>::const_iterator it;
    it = std::find(strVec.begin(), strVec.end(), str);
    // return -1 if str not found in strVec
    if(it==strVec.end())
        return -1;
    return it - strVec.begin();
}

//==============================================================================
//                                  PAPI
//==============================================================================

/// initialise Papi instance
Papi* Papi::instance = NULL;


/**
 * Get papi class instance
 * @return Papi instance
 */
Papi* Papi::Instance()
{
  return instance ? instance : (instance = new Papi);
}

/**
 * Initialise papi
 */
void Papi::Init()
{
    // only initialise if not already initialised
  if (setup)
  {
    return;
  }

  int papiError;

  // set debugging if requested by environment variable
  char *debugStr = getenv("PAPI_DEBUG");
  debug = (debugStr != NULL);
  if (debug) 
  {
    std::cerr << "Papi debug mode on" << std::endl;
  }

  // Initialise the papi library */
  papiError = PAPI_library_init(PAPI_VER_CURRENT);
  if (papiError != PAPI_VER_CURRENT)
  {
    std::cerr << "PAPI library init error!" << std::endl;
    exit (1);
  }
  
  #ifdef _OPENMP
    // assume fixed thread affinity, otherwise this approach fails
    papiError = PAPI_thread_init((long unsigned int (*)()) omp_get_thread_num);
    if (papiError != PAPI_OK)
    {
      std::cerr << "Could not initialize the library with openmp."
              << std::endl;
      exit (1);
    }
    numThreads = omp_get_max_threads();
  #else
    numThreads = 1;
  #endif

  threadTime.resize(numThreads);

  // determine the number of hardware counters
  int numHWCounters;
  papiError = numHWCounters = PAPI_num_counters();
  if (papiError <= PAPI_OK)
  {
    std::cerr << "PAPI error : unable to determine number of hardware counters" << std::endl;
    papi_print_error (papiError);
    exit (1);
  }
  if (debug)
  {
    std::cout << "There are " << numHWCounters
            << " hardware counters available" << std::endl;
  }

    // get user-defined list of hardware counters from environment variable
  char *papiCounters = getenv("PAPI_EVENTS");
  if (debug)
  {
    std::cout << "PAPI_EVENTS = " << papiCounters << std::endl;
  }

  char *result = NULL;
  char delim[] = "|";

  result = strtok(papiCounters, delim);
  while (result != NULL)
  {
    int eventID;
    papiError = PAPI_event_name_to_code(result, &eventID);
    if (papiError == PAPI_OK
        && std::find(events.begin(), events.end(), eventID) == events.end())
    {
      eventNames.push_back(std::string(result));
      events.push_back(eventID);
    }
    else
    {
      std::cerr << "Papi Error : not adding event : " << result << std::endl;
    }
    result = strtok(NULL, delim);
  }
  if (debug)
  {
    std::cout << "there are " << eventNames.size()
            << " requested counters" << std::endl;
  }

  if (GetNumberOfEvents() == 0)
  {
    setup = true;
    return;
  }

  if (GetNumberOfEvents() > 127)
  {
    std::cerr << "Too many events selected : exiting" << std::endl;
    exit(-1);
  }

  eventSet = PAPI_NULL;
  papiError = PAPI_create_eventset(&eventSet);

  if (papiError != PAPI_OK)
  {
    std::cerr << "Papi error : Could not create the EventSet" << std::endl;
    papi_print_error(papiError);
    exit(-1);
  }

  if (debug)
  {
    for (int i = 0; i < GetNumberOfEvents(); i++)
      std::cerr << "Event " << i << " out of " << GetNumberOfEvents()
      << " = " << GetEventName(i) << std::endl;
  }

  // allocate space for counters
  hwCounterValues.resize(numThreads);
  for (int i = 0; i < numThreads; i++)
  {
    hwCounterValues[i].resize(GetNumberOfEvents());
  }

  setup = true;
}


/**
 * Print PAPI error
 * @param  [in] papiErrorCode 
 */
void Papi::papi_print_error(const int papiErrorCode) const
{
  char * errString = PAPI_strerror(papiErrorCode);
  std::cerr << "PAPI error : " << errString << std::endl;
}

/**
 * Start PAPI counters
 */
void Papi::StartCounters()
{
  if (!setup)
  {
    Init();
  }
  
  if (IsCounting())
  {
    std::cerr << "PAPI counters error : cannot start papi counters when they are already running"
            << std::endl;
    exit(-1);
  }

#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    if (GetNumberOfEvents())
    {
      int papiError = PAPI_start_counters(&events[0], events.size());
      if (papiError != PAPI_OK)
      {
        std::cerr << "PAPI error : unable to start counters" << std::endl;
        papi_print_error(papiError);
        exit(-1);
      }
    }

#ifdef _OPENMP
    int threadIndex = omp_get_thread_num();
    double timeTmp = omp_get_wtime();
#else
    int threadIndex = 0;
    double timeTmp = PAPI_get_virt_usec() / 1e6;

#endif
    threadTime[threadIndex] = -timeTmp;
  }
  counting = true;
}


/**
 * Stop PAPI counters
 */
void Papi::StopCounters()
{
  if (!IsCounting())
  {
    std::cerr << "PAPI counters error : cannot stop papi counters when they are have not been started" << std::endl;
    exit(-1);
  }
  
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int threadIndex = omp_get_thread_num();
#else
    int threadIndex = 0;
#endif
    if (GetNumberOfEvents())
    {
      int papiError = PAPI_stop_counters(&hwCounterValues[threadIndex][0], events.size());
      if (papiError != PAPI_OK)
      {
        std::cerr << "PAPI error : unable to stop counters" << std::endl;
        papi_print_error(papiError);
        exit(-1);
      }
    }

#ifdef _OPENMP
    threadTime[threadIndex] += omp_get_wtime();
#else
    threadTime[threadIndex] += (PAPI_get_virt_usec() / 1e6);
#endif                
  }
  counting = false;
}

//==============================================================================
//                              PapiCounter
//==============================================================================
/**
 * Constructor
 */
PapiCounter::PapiCounter()
{
  Papi::Instance()->Init();

  const int numCounters = Papi::Instance()->GetNumberOfEvents();
  const int numThreads = Papi::Instance()->GetNumThreads();
  
  for (int i = 0; i < numCounters; i++)
  {
    names.push_back(Papi::Instance()->GetEventName(i));
    numbers.push_back(Papi::Instance()->GetEventNumber(i));
  }  
  
  counterValues.resize(numThreads);    
  for (int tid = 0; tid < numThreads; tid++)
  {
    counterValues[tid].resize(numCounters, 0LL);
  }
  times.resize(numThreads);
}

/**
 * Stop counters (accumulate)
 */
void PapiCounter::Stop()
{
  Papi::Instance()->StopCounters();
  
  const int numCounters = Papi::Instance()->GetNumberOfEvents();
  const int numThreads = Papi::Instance()->GetNumThreads();
  
  for (int tid = 0; tid < numThreads; tid++)
  {
    for (int i = 0; i < numCounters; i++)
    {
      counterValues[tid][i] += Papi::Instance()->GetCounter(tid, i);
    }
    times[tid] += Papi::Instance()->GetTime(tid);
  }
}

/**
 * Start counter
 */
void PapiCounter::Start()
{
  Papi::Instance()->StartCounters();
}

/**
 * Get aggregated values over all threads
 * @param i - index of the counter
 * @return value for the counter over all threads
 */
long long PapiCounter::GetAggregaterdCounterValuesOverAllThreads(const int i) const
{
  assert(i < GetNumCounters());
  long long sum = 0LL;

  for (int tid = 0; tid < GetNumThreads(); tid++)
  {
    sum += GetValue(tid, i);
  //return sum/(long long)threads();
  }
  return sum;
}

/**
 * Get counter values for each thread 
 * @param counter index
 * @return vector with individual thread values for the counter
 */
std::vector<long long> PapiCounter::GetIndividualValues(const int i) const
{
  assert(i < GetNumCounters());  
  std::vector<long long> tmp;
  
  for (int tid = 0; tid < GetNumThreads(); tid++) 
  {
    tmp.push_back(GetValue(tid, i));
  }
  return tmp;
}
/**
 * Write counter to file
 * @param runName 
 * @param eventID
 * @param fileId
 * @param fileFormat
 */
void PapiCounter::WriteToStream(std::string const &routineName, int eventId, std::ofstream &stream, PapiFileFormat fileFormat)
{
  if (GetNumCounters())
  {
    int numThreads = Papi::Instance()->GetNumThreads();
    
    switch (fileFormat)
    {
      case FileFormatPlain:
        stream << "----------------------------" << std::endl;
        stream << routineName << " :: wall time " << GetTime() << " s" << std::endl;
        stream << "----------------------------" << std::endl;
      
        if (Papi::Instance()->GetNumThreads() > 1)
        {
          for (int tid = 0; tid < Papi::Instance()->GetNumThreads(); tid++)
          {
            stream << "     THREAD" << std::setw(2) << tid;
          }
        }
        stream << " [        TOTAL ]" << std::endl;
        
        for (int i = 0; i < GetNumCounters(); i++)
        {
          if (Papi::Instance()->GetNumThreads() > 1)
          {
            for (int tid = 0; tid < Papi::Instance()->GetNumThreads(); tid++)
            {
              stream << " " << std::setw(12) << GetValue(tid, i);
            }
          }
          stream << " [ " << std::setw(12) << GetAggregaterdCounterValuesOverAllThreads(i) << " ]"
                  << "\t" << GetName(i) << std::endl;
        }
        /*
        for(int i in derivedCounters<platform>::counters){
            if( derivedCounters<platform>::counters[i].is_available() )
                compute;
                write to screen;
        }
         */
        
        
        if (IsDerivedStatAvailable(Derived_FLIPS))
        {
          std::vector<double> stat = ComputederivedStat(Derived_FLIPS);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {
              stream << " " << std::setw(12) << stat[tid] / VectorMean(times) / (1.e6);
            }
          }
          stream << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6)<< " ]"
                  << "\tderived_FLIPS (MFLIPS)" << std::endl;
        }
        
        if (IsDerivedStatAvailable(Derived_FLOPS))
        {
          std::vector<double> stat = ComputederivedStat(Derived_FLOPS);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {
              stream << " " << std::setw(12) << stat[tid] / VectorMean(times) / (1.e6);
            }
          }
          stream << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6)<< " ]"
                  << "\tderived_FLOPS (MFLOPS)" << std::endl;
        }


        if (IsDerivedStatAvailable(Derived_DP_vector_FLOPS))
        {
          std::vector<double> stat = ComputederivedStat(Derived_DP_vector_FLOPS);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {
              stream << " " << std::setw(12) << stat[tid] / VectorMean(times) / (1.e6);
            }
          }
          stream << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6)<< " ]"
                  << "\tderived_DP_vector_FLOPS (MFLOPS)" << std::endl;
        }
        
        if (IsDerivedStatAvailable(Derived_SP_vector_FLOPS))
        {
          std::vector<double> stat = ComputederivedStat(Derived_SP_vector_FLOPS);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {
              stream << " " << std::setw(12) << stat[tid] / VectorMean(times) / (1.e6);
            }
          }
          stream << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6)<< " ]"
                  << "\tderived_SP_vector_FLOPS (MFLOPS)" << std::endl;
        }


        if (IsDerivedStatAvailable(Derived_L1_DMR))
        {
          std::vector<double> stat = ComputederivedStat(Derived_L1_DMR);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {
              stream << std::setw(12) << std::setprecision(3)<< stat[tid] * 100 << "%";
            }           
            stream << " [ " << std::setw(12) << "-" << " ]" << "\tderived_L1_DMR (%)" << std::endl;
          } else {
            stream << " [ " << std::setw(11) << std::setprecision(3)<< stat[0] * 100 << "%" << " ]" << "\tderived_L1_DMR (%)" << std::endl;              
          }          
        }

        if (IsDerivedStatAvailable(Derived_L2_DMR))
        {
          std::vector<double> stat = ComputederivedStat(Derived_L2_DMR);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {
              stream << std::setw(12) << std::setprecision(3)<< stat[tid] * 100 << "%";
            }          
          stream<< " [ " << std::setw(12) << "-" << " ]" << "\tderived_L2_DMR (%)" << std::endl;
          } else {
            stream << " [ " << std::setw(11) << std::setprecision(3)<< stat[0] * 100 << "%" << " ]" << "\tderived_L2_DMR (%)" << std::endl;              
          }
          
        }

         if (IsDerivedStatAvailable(Derived_L1_TMR))
        {
          std::vector<double> stat = ComputederivedStat(Derived_L1_TMR);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {
              stream << std::setw(12) << std::setprecision(3)<< stat[tid] * 100 << "%";
            }          
          stream << " [ " << std::setw(12) << "-" << " ]" << "\tderived_L1_TMR (%)" << std::endl;
          } else {
            stream << " [ " << std::setw(11) << std::setprecision(3)<< stat[0] * 100 << "%" << " ]" << "\tderived_L1_TMR (%)" << std::endl;              
          }
          
        }

        if (IsDerivedStatAvailable(Derived_L2_TMR))
        {
          std::vector<double> stat = ComputederivedStat(Derived_L2_TMR);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {
              stream << std::setw(12) << std::setprecision(3)<< stat[tid] * 100 << "%";
            }          
          stream<< " [ " << std::setw(12) << "-" << " ]" << "\tderived_L2_TMR (%)" << std::endl;
          } else {
            stream << " [ " << std::setw(11) << std::setprecision(3)<< stat[0] * 100 << "%" << " ]" << "\tderived_L2_TMR (%)" << std::endl;              
          }          
        }

        if (IsDerivedStatAvailable(Derived_L3_TMR))
        {
          std::vector<double> stat = ComputederivedStat(Derived_L3_TMR);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {
              stream << std::setw(12) << std::setprecision(3)<< stat[tid] * 100 << "%";
            }          
          stream<< " [ " << std::setw(12) << "-" << " ]" << "\tderived_L3_TMR (%)" << std::endl;
          } else {
            stream << " [ " << std::setw(11) << std::setprecision(3)<< stat[0] * 100 << "%" << " ]" << "\tderived_L3_TMR (%)" << std::endl;              
          }          
        }
        
        if (IsDerivedStatAvailable(Derived_Mem_Bandwidth))
        {
          std::vector<double> stat = ComputederivedStat(Derived_Mem_Bandwidth);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {                                
              stream << std::setw(10) << std::setprecision(3)<< VectorSum(stat) * 64 / VectorMean(times) / (1024*1024)<< "MB/s";
            }          
          stream<< " [ " << std::setw(10) << "-" << " ]" << "\tderived_Mem_Bandwidth [MB/s]" << std::endl;
          } else {
            stream << " [ " << std::setw(9) << std::setprecision(3)<< VectorSum(stat) * 64 / VectorMean(times) / (1024*1024)<< "MB/s"<< " ]" << "\tderived_Mem_Bandwidth [MB/s]" << std::endl;              
          }          
        }
                        
        
        if (IsDerivedStatAvailable(Derived_BANDWIDTH_SS))
        {
          std::vector<double> stat = ComputederivedStat(Derived_BANDWIDTH_SS);
          if (numThreads > 1)
            for (int tid = 0; tid < numThreads; tid++)
              stream << "     -     ";
          stream << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6) << " ]"
                  << "\tderived_BANDWIDTH_SS (MB/s)" << std::endl;
        }
        
        if (IsDerivedStatAvailable(Derived_BANDWIDTH_DS))
        {
          std::vector<double> stat = ComputederivedStat(Derived_BANDWIDTH_DS);
          if (numThreads > 1)
            for (int tid = 0; tid < numThreads; tid++)
              stream << "     -     ";
          stream << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6) << " ]"
                  << "\tderived_BANDWIDTH_DS (MB/s)" << std::endl;
        }

        break;
      case FileFormatMatlab:
        stream << routineName << " = " << eventId << ";" << std::endl;
        for (int i = 0; i < GetNumCounters(); i++)
        {
          std::stringstream vname;
          vname << "event{" <<routineName<< "}.counter(" << i + 1 << ").count";
          stream << "event{" <<routineName<< "}.counter(" << i + 1 << ").name = \'"
                  << GetName(i) << "\';" << std::endl;
          writeVecMatlab(stream, vname.str(), GetIndividualValues(i));
        }
        break;
        
      case FileFormatLaTeX:
        stream << "\\hline" << std::endl;
        stream << "\\multicolumn{2}{c}{" << routineName << "}" << std::endl;
        stream << "\\hline" << std::endl;
        stream << "counter & count" << "\\\\" << std::endl;
        stream << "\\hline" << std::endl;
        for (int i = 0; i < GetNumCounters(); i++)
          stream << "\\lst{" << GetName(i) << "}"
          << " & " << GetAggregaterdCounterValuesOverAllThreads(i) << "\\\\" << std::endl;
        break;
    }
  }
}

/**
 * Print to screan
 */
void PapiCounter::PrintScreen()
{
  int numThreads = Papi::Instance()->GetNumThreads();
  
  if (GetNumCounters() > 0)
  {
    if (Papi::Instance()->GetNumThreads() > 1)
    {
      for (int tid = 0; tid < Papi::Instance()->GetNumThreads(); tid++)
      {
        std::cout << "     THREAD" << std::setw(2) << tid;
      }
    }
    
    std::cout << " [        TOTAL ]" << std::endl;
    for (int i = 0; i < GetNumCounters(); i++)
    {
      if (Papi::Instance()->GetNumThreads() > 1)
      {
        for (int tid = 0; tid < Papi::Instance()->GetNumThreads(); tid++)
        {
          std::cout << " " << std::setw(12) << GetValue(tid, i);
        }
      }
      std::cout << " [ " << std::setw(12) << GetAggregaterdCounterValuesOverAllThreads(i) << " ]"
              << "\t" << GetName(i) << std::endl;
    }
    

    if (IsDerivedStatAvailable(Derived_FLIPS))
    {
      std::vector<double> stat = ComputederivedStat(Derived_FLIPS);
      if (numThreads > 1)
      {
        for (int tid = 0; tid < numThreads; tid++)
        {
          std::cout << " " << std::setw(12) << stat[tid] / VectorMean(times) / (1.e6);
        }
      }
      std::cout << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6) << " ]"
              << "\tderived_FLIPS (MFLIPS)" << std::endl;
    }

    if (IsDerivedStatAvailable(Derived_FLOPS))
    {
      std::vector<double> stat = ComputederivedStat(Derived_FLOPS);
      if (numThreads > 1)
      {
        for (int tid = 0; tid < numThreads; tid++)
        {
          std::cout << " " << std::setw(12) << stat[tid] / VectorMean(times) / (1.e6);
        }
      }
      std::cout << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6) << " ]"
              << "\tderived_FLOPS (MFLOPS)" << std::endl;
    }
    
    if (IsDerivedStatAvailable(Derived_DP_vector_FLOPS))
    {
      std::vector<double> stat = ComputederivedStat(Derived_DP_vector_FLOPS);
      if (numThreads > 1)
      {
        for (int tid = 0; tid < numThreads; tid++)
        {
          std::cout << " " << std::setw(12) << stat[tid] / VectorMean(times) / (1.e6);
        }
      }
      std::cout << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6) << " ]"
              << "\tderived_DP_vector_FLOPS (MFLOPS)" << std::endl;
    }
    
    if (IsDerivedStatAvailable(Derived_SP_vector_FLOPS))
    {
      std::vector<double> stat = ComputederivedStat(Derived_SP_vector_FLOPS);
      if (numThreads > 1)
      {
        for (int tid = 0; tid < numThreads; tid++)
        {
          std::cout << " " << std::setw(12) << stat[tid] / VectorMean(times) / (1.e6);
        }
      }
      std::cout << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6) << " ]"
              << "\tderived_SP_vector_FLOPS (MFLOPS)" << std::endl;
    }
    
    if (IsDerivedStatAvailable(Derived_L1_DMR))
    {
      std::vector<double> stat = ComputederivedStat(Derived_L1_DMR);
      if (numThreads > 1)
      {
        for (int tid = 0; tid < numThreads; tid++)
        {
          std::cout << std::setw(12) << std::setprecision(3)<< stat[tid] * 100 << "%";
        }
        std::cout << " [ " << std::setw(12) << "-" << " ]" << "\tderived_L1_DMR (%)" << std::endl;
      } else {
        std::cout << " [ " << std::setw(11) << std::setprecision(3)<< stat[0] * 100 << "%" << " ]" << "\tderived_L1_DMR (%)" << std::endl;              
      }
    }
    
    if (IsDerivedStatAvailable(Derived_L2_DMR))
    {
      std::vector<double> stat = ComputederivedStat(Derived_L2_DMR);
      if (numThreads > 1)
      {
        for (int tid = 0; tid < numThreads; tid++)
        {
          std::cout << std::setw(12) << std::setprecision(3)<< stat[tid] * 100 << "%";
        }      
      std::cout << " [ " << std::setw(12) << "-" << " ]" << "\tderived_L2_DMR (%)" << std::endl;
      } else {
        std::cout << " [ " << std::setw(11) << std::setprecision(3)<< stat[0] * 100 << "%" << " ]" << "\tderived_L2_DMR (%)" << std::endl;              
      }
    }
    
    if (IsDerivedStatAvailable(Derived_L1_TMR))
    {
      std::vector<double> stat = ComputederivedStat(Derived_L1_TMR);
      if (numThreads > 1)
      {
        for (int tid = 0; tid < numThreads; tid++)
        {
          std::cout << std::setw(12) << std::setprecision(3)<< stat[tid] * 100 << "%";
        }
        std::cout << " [ " << std::setw(12) << "-" << " ]" << "\tderived_L1_TMR (%)" << std::endl;
      } else {
        std::cout << " [ " << std::setw(11) << std::setprecision(3)<< stat[0] * 100 << "%" << " ]" << "\tderived_L1_TMR (%)" << std::endl;              
      }
            
    }
    
    if (IsDerivedStatAvailable(Derived_L2_TMR))
    {
      std::vector<double> stat = ComputederivedStat(Derived_L2_TMR);
      if (numThreads > 1)
      {
        for (int tid = 0; tid < numThreads; tid++)
        {
          std::cout << std::setw(12) << std::setprecision(3)<< stat[tid] * 100 << "%";
        }      
      std::cout << " [ " << std::setw(12) << "-" << " ]" << "\tderived_L2_TMR (%)" << std::endl;
      } else {
        std::cout << " [ " << std::setw(11) << std::setprecision(3)<< stat[0] * 100 << "%" << " ]" << "\tderived_L2_TMR (%)" << std::endl;              
      }
    }

    if (IsDerivedStatAvailable(Derived_L3_TMR))
    {
      std::vector<double> stat = ComputederivedStat(Derived_L3_TMR);
      if (numThreads > 1)
      {
        for (int tid = 0; tid < numThreads; tid++)
        {
          std::cout << std::setw(12) << std::setprecision(3)<< stat[tid] * 100 << "%";
        }      
      std::cout << " [ " << std::setw(12) << "-" << " ]" << "\tderived_L3_TMR (%)" << std::endl;
      } else {
        std::cout << " [ " << std::setw(11) << std::setprecision(3)<< stat[0] * 100 << "%" << " ]" << "\tderived_L2_TMR (%)" << std::endl;              
      }
    }

    if (IsDerivedStatAvailable(Derived_Mem_Bandwidth))
        {
          std::vector<double> stat = ComputederivedStat(Derived_Mem_Bandwidth);
          if (numThreads > 1)
          {
            for (int tid = 0; tid < numThreads; tid++)
            {                                
              std::cout << std::setw(10) << std::setprecision(3)<< VectorSum(stat) * 64 / VectorMean(times) / (1024*1024)<< "MB/s";
            }          
          std::cout<< " [ " << std::setw(12) << "-" << " ]" << "\tderived_Mem_Bandwidth [MB/s]" << std::endl;
          } else {
            std::cout << " [ " << std::setw(9) << std::setprecision(3)<< VectorSum(stat) * 64 / VectorMean(times) / (1024*1024)<< "MB/s"<< " ]" << "\tderived_Mem_Bandwidth [MB/s]" << std::endl;              
          }          
    }
    
    if (IsDerivedStatAvailable(Derived_BANDWIDTH_SS))
    {
      std::vector<double> stat = ComputederivedStat(Derived_BANDWIDTH_SS);
      if (numThreads > 1)
        for (int tid = 0; tid < numThreads; tid++)
          std::cout << "     -     ";
      std::cout << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6) << " ]"
              << "\tderived_BANDWIDTH_SS (MB/s)" << std::endl;
    }
    
    if (IsDerivedStatAvailable(Derived_BANDWIDTH_DS))
    {
      std::vector<double> stat = ComputederivedStat(Derived_BANDWIDTH_DS);
      if (numThreads > 1)
        for (int tid = 0; tid < numThreads; tid++)
          std::cout << "     -     ";
      std::cout << " [ " << std::setw(12) << VectorSum(stat) / VectorMean(times) / (1.e6) << " ]"
              << "\tderived_BANDWIDTH_DS (MB/s)" << std::endl;
    }
  }
  else
  {
    std::cout << "PAPI-WRAP :: no counters to print" << std::endl;
  }
}

/**
 * Are derived statistics available
 * @param statIdx
 * @return 
 */
bool PapiCounter::IsDerivedStatAvailable(const DerivedStatistics statIdx) const
{
   
  switch (statIdx)
  {
    case Derived_FLIPS:
      return findString(names, std::string("PAPI_FP_INS")) >= 0 ? true : false;
    case Derived_FLOPS:
      return findString(names, std::string("PAPI_FP_OPS")) >= 0 ? true : false;
    case Derived_DP_vector_FLOPS:
      return findString(names, std::string("PAPI_DP_OPS")) >= 0 ? true : false;
    case Derived_SP_vector_FLOPS:
      return findString(names, std::string("PAPI_SP_OPS")) >= 0 ? true : false;
    case Derived_L1_DMR:
      return (
              findString(names, std::string("PAPI_L1_DCA")) >= 0 &&
              findString(names, std::string("PAPI_L1_DCM")) >= 0
              );
    case Derived_L2_DMR:
      return (
              findString(names, std::string("PAPI_L2_DCA")) >= 0 &&
              findString(names, std::string("PAPI_L2_DCM")) >= 0
              );
    case Derived_L1_TMR:
      return (
              findString(names, std::string("PAPI_L1_TCA")) >= 0 &&
              findString(names, std::string("PAPI_L1_TCM")) >= 0
              );
    case Derived_L2_TMR:
      return (
              findString(names, std::string("PAPI_L2_TCA")) >= 0 &&
              findString(names, std::string("PAPI_L2_TCM")) >= 0
              );
   case Derived_L3_TMR:
      return (
              findString(names, std::string("PAPI_L3_TCA")) >= 0 &&
              findString(names, std::string("PAPI_L3_TCM")) >= 0
              );
      
   case Derived_Mem_Bandwidth:
      return findString(names, std::string("PAPI_L3_TCM")) >= 0 ? true : false;   
      
    case Derived_BANDWIDTH_SS:
      return (
              findString(names, std::string("SYSTEM_READ_RESPONSES:0x07")) >= 0 &&
              findString(names, std::string("OCTWORD_WRITE_TRANSFERS:0x01")) >= 0
              );
    case Derived_BANDWIDTH_DS:
      return (
              findString(names, std::string("SYSTEM_READ_RESPONSES:0x07")) >= 0 &&
              findString(names, std::string("OCTWORD_WRITE_TRANSFERS:0x01")) >= 0
              );
  }
  return false;
}
/**
 * Compute derived statistics
 * @param statIdx 
 * @return 
 */
std::vector<double> PapiCounter::ComputederivedStat(DerivedStatistics statIdx)
{
  std::vector<double> derived(GetNumThreads());
  int idx, idxCM, idxCA;
  int idxSRS, idxOWT;
  
  switch (statIdx)
  {
    case Derived_FLIPS:
      idx = findString(names, std::string("PAPI_FP_INS"));
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        // returns total number of FP operations
        // determine FLOPS by summing and dividing by time
        derived[tid] = GetValue(tid, idx);
      }
      return derived;      
      
    case Derived_FLOPS:
      idx = findString(names, std::string("PAPI_FP_OPS"));
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        // returns total number of FP operations
        // determine FLOPS by summing and dividing by time
        derived[tid] = GetValue(tid, idx);
      }
      return derived;      

    case Derived_DP_vector_FLOPS:
      idx = findString(names, std::string("PAPI_DP_OPS"));
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        // returns total number of FP operations
        // determine FLOPS by summing and dividing by time
        derived[tid] = GetValue(tid, idx);
      }
      return derived;      
    
    case Derived_SP_vector_FLOPS:
      idx = findString(names, std::string("PAPI_SP_OPS"));
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        // returns total number of FP operations
        // determine FLOPS by summing and dividing by time
        derived[tid] = GetValue(tid, idx);
      }
      return derived;      
      
    case Derived_L1_DMR:
      idxCM = findString(names, std::string("PAPI_L1_DCM"));
      idxCA = findString(names, std::string("PAPI_L1_DCA"));
      
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        derived[tid] = (double) GetValue(tid, idxCM) / (double) GetValue(tid, idxCA);
      }
      return derived;
      
    case Derived_L2_DMR:
      idxCM = findString(names, std::string("PAPI_L2_DCM"));
      idxCA = findString(names, std::string("PAPI_L2_DCA"));
      
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        derived[tid] = (double) GetValue(tid, idxCM) / (double) GetValue(tid, idxCA);
      }
      return derived;
      
    case Derived_L1_TMR:
      idxCM = findString(names, std::string("PAPI_L1_TCM"));
      idxCA = findString(names, std::string("PAPI_L1_TCA"));
      
      for (int tid = 0; tid < GetNumThreads(); tid++) 
      {
        derived[tid] = (double) GetValue(tid, idxCM) / (double) GetValue(tid, idxCA);
      }
      return derived;
      
    case Derived_L2_TMR:
      idxCM = findString(names, std::string("PAPI_L2_TCM"));
      idxCA = findString(names, std::string("PAPI_L2_TCA"));
      
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        derived[tid] = (double) GetValue(tid, idxCM) / (double) GetValue(tid, idxCA);
      }
      return derived;
      
    case Derived_L3_TMR:
      idxCM = findString(names, std::string("PAPI_L3_TCM"));
      idxCA = findString(names, std::string("PAPI_L3_TCA"));
      
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        derived[tid] = (double) GetValue(tid, idxCM) / (double) GetValue(tid, idxCA);
      }
      return derived;
      
            
    case Derived_Mem_Bandwidth:
      idx = findString(names, std::string("PAPI_L3_TCM"));
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        // returns total number of FP operations
        // determine FLOPS by summing and dividing by time
        derived[tid] = GetValue(tid, idx);
      }
      return derived;
      
      // see page 2101 of Shirley Moore et al., Procedia Computer Science 4 (2011)
      // this needs some tweaking for Interlagos (15h):
    case Derived_BANDWIDTH_SS:
      idxSRS = findString(names, std::string("SYSTEM_READ_RESPONSES:0x07"));
      idxOWT = findString(names, std::string("OCTWORD_WRITE_TRANSFERS:0x01"));
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        derived[tid] = (double) GetValue(tid, idxSRS)*32. + (double) GetValue(tid, idxOWT)*8.;
      }      
      return derived;
    case Derived_BANDWIDTH_DS:
      idxSRS = findString(names, std::string("SYSTEM_READ_RESPONSES:0x07"));
      idxOWT = findString(names, std::string("OCTWORD_WRITE_TRANSFERS:0x01"));
      // dual stream: accumulate bandwidth of one core per Bulldozer module
      for (int tid = 0; tid < GetNumThreads(); tid++)
      {
        derived[tid] = (double) GetValue(tid, idxSRS)*32. + (double) GetValue(tid, idxOWT)*4.;
      }
      return derived;
  }
  // this will never occur - just to keep the compiler warnings off
  return std::vector<double>(0);
}

/*================================================
            PapiCounterList
================================================*/
/**
 * Add routine to the papi couner
 * @param routineName
 */
void PapiCounterList::AddRoutine(const std::string routineName)
{
  // ensure that someone hasn't already added an event with this name
  assert(routineEvents.find(routineName) == routineEvents.end());

  routineEvents[routineName] = PapiCounter();
}
/**
 * Get counter for the routine
 * @param routineName
 * @return return counter for the routine number
 */
PapiCounter& PapiCounterList::Routine(std::string routineName)
{
  // ensure that an event with ename exists
  assert(routineEvents.find(routineName) != routineEvents.end());

  return routineEvents[routineName];
}

/**
 * Write to stram
 * @param fileName
 * @param fileFormat
 */
void PapiCounterList::WriteToFile(const std::string fileName, PapiFileFormat fileFormat)
{
  std::ofstream fid;
  fid.open(fileName.c_str());

  switch (fileFormat)
  {
    case FileFormatMatlab:
      break;
    case FileFormatPlain:
      break;
    case FileFormatLaTeX:
      fid << "\\begin{tabular}{lr}" << std::endl;
      break;
  }

  int id = 1;
  for (std::map<std::string, PapiCounter>::iterator it = routineEvents.begin();
          it != routineEvents.end();
          it++)
  {
    it->second.WriteToStream(it->first, id, fid, fileFormat);
    id++;
  }

  switch (fileFormat)
  {
    case FileFormatMatlab:
      break;
    case FileFormatPlain:
      break;
    case FileFormatLaTeX:
      fid << "\\hline" << std::endl;
      fid << "\\end{tabular}" << std::endl;
      break;
  }

  fid.close();
}
/**
 * Write to stream
 * @param fid
 * @param fileFormat
 */
void PapiCounterList::WriteToFile(std::ofstream &fstream, PapiFileFormat fileFormat)
{
  switch (fileFormat)
  {
    case FileFormatMatlab:
      break;
    case FileFormatPlain:
      break;
    case FileFormatLaTeX:
      fstream << "\\begin{tabular}{lr}" << std::endl;
      break;
  }

  int id = 1;
  for (std::map<std::string, PapiCounter>::iterator it = routineEvents.begin();
          it != routineEvents.end();
          it++)
  {
    it->second.WriteToStream(it->first, id++, fstream, fileFormat);
  }

  switch (fileFormat)
  {
    case FileFormatMatlab:
      break;
    case FileFormatPlain:
      break;
    case FileFormatLaTeX:
      fstream << "\\hline" << std::endl;
      fstream << "\\end{tabular}" << std::endl;
      break;
  }

  // close the file stream
  fstream.close();
}
/**
 * Print to screen
 */
void PapiCounterList::PrintScreen()
{
  for (std::map<std::string, PapiCounter>::iterator it = routineEvents.begin();
          it != routineEvents.end();
          it++)
  {
    std::cout << "--------------------------------" << std::endl;
    std::cout << it->first << " :: wall time " << it->second.GetTime() << " s" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    it->second.PrintScreen();
  }
}
