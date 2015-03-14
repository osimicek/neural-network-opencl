/**
Copyright (c) 2012, Swiss National Supercomputing Center (CSCS)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the Swiss National Supercomputing Center (CSCS) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef PAPI_COUNTER_H
#define	PAPI_COUNTER_H



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

/**
 * @enum DerivedStatistics
 * @brief Derived PAPI statistics
 */
enum DerivedStatistics{
    Derived_FLIPS,
    Derived_FLOPS,
    Derived_DP_vector_FLOPS,
    Derived_SP_vector_FLOPS,
    Derived_L1_DMR,
    Derived_L2_DMR,
    Derived_L1_TMR,
    Derived_L2_TMR,
    Derived_L3_TMR,
    Derived_Mem_Bandwidth,
    Derived_BANDWIDTH_SS,
    Derived_BANDWIDTH_DS
};

/**
 * Get sum of the vector
 * @param [in] v - input vector
 * @return sum of the vector elements
 */
template <typename T>
T VectorSum(std::vector<T> const &v)
{
    T sum = T();
    for(size_t i=0; i<v.size(); i++)
    {
        sum += v[i];
    }
    return sum;
}

/**
 * Get mean value of the vector
 * @param v
 * @return 
 */
template <typename T>
T VectorMean(std::vector<T> const &v)
{
    return VectorSum(v)/(T)v.size();
}

/**
 * Write vector in Matlab format
 * @param fid  - file to write into
 * @param name - Name of the vector (measured routine)
 * @param v    - input vector
 */
template <typename TVec>
void writeVecMatlab(std::ofstream &fid, const std::string & name, TVec const &v)
{
    fid << name << " = [";
    for(size_t i=0; i<v.size(); i++)
    {
        fid << v[i] << (i<(v.size()-1) ? " " : "];");
    }
    fid << std::endl;
}

/**
 * @enum PapiFileFormat
 * @brief Enumerate the different output formats for counter information \n
 *        LaTex support not currently implemented
 */
enum PapiFileFormat {FileFormatMatlab, FileFormatPlain, FileFormatLaTeX};


/**
 * @class Papi
 * @brief singleton that handles papi intialisation and access to papi calls
 */
class Papi
{
  public:
    /// Get papi class instance
    static Papi* Instance();
    /// destructor
    ~Papi();
    /// Initialise papi
    void Init();
    
    /// Get Numbero of PAPI events
    inline int GetNumberOfEvents() const
    {
        return eventNames.size();
    };
    
    /// Get event name
    std::string const &GetEventName(const int eventIndex) const
    {
      assert(eventIndex<GetNumberOfEvents());
      return eventNames[eventIndex];
    };
    
    /// Get event number
    int GetEventNumber(const int eventIndex) const
    {
      assert(eventIndex<GetNumberOfEvents());
      return events[eventIndex];
    };
    
    /// Get counter for a given thread
    long long GetCounter(const int threadIdx, const int counterIndex) const
    {
      assert(counterIndex<GetNumberOfEvents());
      assert(threadIdx<GetNumThreads());
      return hwCounterValues[threadIdx][counterIndex];
    };
    
    /// Get time for given thread
    double GetTime(int ThreadIdx) const
    {
      assert(ThreadIdx<GetNumThreads());
      return threadTime[ThreadIdx];
    };
    
    /// Start PAPI counters
    void StartCounters();
    /// Stop PAPI counters
    void StopCounters();
    /// Get number of threads
    int GetNumThreads() const 
    {
      return numThreads;
    };
    
    /// Is counting?
    bool IsCounting() const 
    { 
      return counting; 
    };
  private:
    /// Default constructor  
    Papi() : setup(false), debug(false), counting(false) {}; 
    /// COPY constructor
    Papi(Papi const &) {};
    
    /// Print papi error
    void papi_print_error(const int papiErrorCode) const;

    bool setup;
    bool debug;
    bool counting;
    int eventSet;
    int numThreads;
    std::vector<std::string> eventNames;
    std::vector<int> events;
    std::vector<double> threadTime;
    /// actual counter HW counter values
    std::vector<std::vector<long long> > hwCounterValues;

    static Papi* instance;
};

/**
 * @class PapiCounter 
 * @brief Class with counters for given routine
 */
class PapiCounter
{
  public:
    /// Constructor
    PapiCounter();
    /// Start counters
    void Start();
    /// Stop counters
    void Stop();
    /// Write to file
    void WriteToStream(std::string const &routineName,
                     int eventId,
                     std::ofstream &stream,
                     const PapiFileFormat fileFormat);
    /// Get name
    std::string GetName(const int i) const
    {
      return names[i];
    };
    
    /// Get number
    int GetNumber(const int i) const
    {
      return numbers[i];
    };
    
    /// Get count
    long long GetValue(const int threadIdx, const int i) const
    {
      assert(threadIdx < GetNumThreads());
      return counterValues[threadIdx][i];
    };
    
    /// Get time for thread
    double GetTime(const int threadIdx) const
    {
      assert(threadIdx < GetNumThreads());
      return times[threadIdx];
    };
    
    /// Get aggregated time
    double GetTime() const
    {
      return VectorMean(times);
    };
    
    /// Get number of counters
    int GetNumCounters() const
    {
      return names.size();
    };
    
    /// Get number of threads
    int GetNumThreads() const
    {
      return Papi::Instance()->GetNumThreads();
    };
    
    /// Get counters across all threads
    long long GetAggregaterdCounterValuesOverAllThreads(const int i) const;
    /// print Screen
    void PrintScreen();
    
    std::vector<long long> GetIndividualValues(const int i) const;
    
  private:
    /// Is derived statistics available
    bool IsDerivedStatAvailable(const DerivedStatistics statIdx) const;
    /// Compute derived statistics
    std::vector<double> ComputederivedStat(const DerivedStatistics statIdx);
    
    std::vector<std::string> names;
    std::vector<int> numbers;
    std::vector<double> times;
    /// counters for a given routine over multiple invocations
    std::vector<std::vector<long long> > counterValues;

};


/**
 * @class PapiCounterList
 * @brief class to manage all events that we want to benchmark \n
 *        essentially a wrapper around map<string, PapiCoutner> where the string \n
 *        is the routine name or a named code section
 */
class PapiCounterList
{
public:
  /// constructor
  PapiCounterList() { };
  /// write to stream
  void WriteToFile(const std::string fileName, const PapiFileFormat fileFormat = FileFormatPlain);
  /// write to stream
  void WriteToFile(std::ofstream &fstream, const PapiFileFormat fileFormat = FileFormatPlain);
  ///print to screen
  void PrintScreen();
  /// add routine
  void AddRoutine(const std::string routineName);
  /// Routine
  PapiCounter& Routine(const std::string routineName);

  /// override [] to allow access to events using ["eventName"]
  PapiCounter& operator[] (std::string &routineName)
  {
    return Routine(routineName);
  };
  /// operator []
  PapiCounter& operator[] (std::string routineName)
  {
    return Routine(routineName);
  };
private:
  std::map<std::string, PapiCounter> routineEvents;
};

#endif
