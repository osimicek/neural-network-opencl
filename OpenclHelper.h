#ifndef OPENCL_HELPER_H
#define OPENCL_HELPER_H

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
// #include <CL/cl.hpp>
#include "./cl.hpp"
#endif

using namespace std;
using namespace cl;

class OpenclHelper {
    public:
        static const char *cl_error_to_str(cl_int e);
        static void print_platforms_devices(bool full);
        static void print_device_info(Device *device);
        static Device get_device(uint platformIndex, uint deviceIndex);
        static char *read_file(const char *filename);
        static Program build_program(Context ctx, std::vector<std::string> codes, char const *options);
};


/* An error check macro for OpenCL.
 *
 * Usage:
 * CHECK_CL_ERROR(status_code_from_a_cl_operation, "function_name")
 *
 * It will abort with a message if an error occurred.
 */

#define CHECK_CL_ERROR(STATUS_CODE, WHAT) \
    if ((STATUS_CODE) != CL_SUCCESS) \
    { \
        fprintf(stderr, \
            "*** '%s' in '%s' on line %d failed with error '%s'.\n", \
            WHAT, __FILE__, __LINE__, \
            OpenclHelper::cl_error_to_str(STATUS_CODE)); \
        abort(); \
    }

/* A more automated error check macro for OpenCL, for use with clXxxx
 * functions that return status codes. (Not all of them do, notably 
 * clCreateXxx do not.)
 *
 * Usage:
 * CALL_CL_GUARDED(clFunction, (arg1, arg2));
 *
 * Note the slightly strange comma between the function name and the
 * argument list.
 */

#define CALL_CL_GUARDED(NAME, ARGLIST) \
    { \
        cl_int status_code; \
        status_code = NAME ARGLIST; \
        CHECK_CL_ERROR(status_code, #NAME); \
    }

/* An error check macro for Unix system functions. If "COND" is true, then the
 * last system error ("errno") is printed along with MSG, which is supposed to
 * be a string describing what you were doing.
 *
 * Example:
 * CHECK_SYS_ERROR(dave != 0, "opening hatch");
 */
#define CHECK_SYS_ERROR(COND, MSG) \
    if (COND) \
    { \
        perror(MSG); \
        abort(); \
    }

#endif
