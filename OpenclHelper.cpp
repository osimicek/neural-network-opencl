#include <iostream>
#include <string>
#include <stdio.h>
#include <map>
#include "OpenclHelper.h"
/**
 * OpenclHelper was inspired by Andreas Kloeckners implementation of cl-helper in C.
 * https://github.com/alexkaiser/mc_stretch/blob/master/cl-helper.c
 */


/**
 * Prints OpenCL error.
 */
const char *OpenclHelper::cl_error_to_str(cl_int e)
{
  switch (e)
  {
    case CL_SUCCESS: return "success";
    case CL_DEVICE_NOT_FOUND: return "device not found";
    case CL_DEVICE_NOT_AVAILABLE: return "device not available";
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
    case CL_COMPILER_NOT_AVAILABLE: return "device compiler not available";
#endif
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "mem object allocation failure";
    case CL_OUT_OF_RESOURCES: return "out of resources";
    case CL_OUT_OF_HOST_MEMORY: return "out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "profiling info not available";
    case CL_MEM_COPY_OVERLAP: return "mem copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH: return "image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "image format not supported";
    case CL_BUILD_PROGRAM_FAILURE: return "build program failure";
    case CL_MAP_FAILURE: return "map failure";

    case CL_INVALID_VALUE: return "invalid value";
    case CL_INVALID_DEVICE_TYPE: return "invalid device type";
    case CL_INVALID_PLATFORM: return "invalid platform";
    case CL_INVALID_DEVICE: return "invalid device";
    case CL_INVALID_CONTEXT: return "invalid context";
    case CL_INVALID_QUEUE_PROPERTIES: return "invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE: return "invalid command queue";
    case CL_INVALID_HOST_PTR: return "invalid host ptr";
    case CL_INVALID_MEM_OBJECT: return "invalid mem object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE: return "invalid image size";
    case CL_INVALID_SAMPLER: return "invalid sampler";
    case CL_INVALID_BINARY: return "invalid binary";
    case CL_INVALID_BUILD_OPTIONS: return "invalid build options";
    case CL_INVALID_PROGRAM: return "invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "invalid program executable";
    case CL_INVALID_KERNEL_NAME: return "invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION: return "invalid kernel definition";
    case CL_INVALID_KERNEL: return "invalid kernel";
    case CL_INVALID_ARG_INDEX: return "invalid arg index";
    case CL_INVALID_ARG_VALUE: return "invalid arg value";
    case CL_INVALID_ARG_SIZE: return "invalid arg size";
    case CL_INVALID_KERNEL_ARGS: return "invalid kernel args";
    case CL_INVALID_WORK_DIMENSION: return "invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE: return "invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE: return "invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET: return "invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST: return "invalid event wait list";
    case CL_INVALID_EVENT: return "invalid event";
    case CL_INVALID_OPERATION: return "invalid operation";
    case CL_INVALID_GL_OBJECT: return "invalid gl object";
    case CL_INVALID_BUFFER_SIZE: return "invalid buffer size";
    case CL_INVALID_MIP_LEVEL: return "invalid mip level";

#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
    case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR: return "invalid gl sharegroup reference number";
#endif

#ifdef CL_VERSION_1_1
    case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "misaligned sub-buffer offset";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "exec status error for events in wait list";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "invalid global work size";
#endif

    default: return "invalid/unknown error code";
  }
}

/**
 * Prints all informations about device
 */
void OpenclHelper::print_device_info(Device *device) {
    long long val;
    cl_int status;
    // device name
    STRING_CLASS name;
    status = device->getInfo(CL_DEVICE_NAME, &name);
    printf("    name: %s'\n", name.c_str());

    STRING_CLASS vendor;
    status = device->getInfo(CL_DEVICE_VENDOR, &vendor);
    printf("    vendor: %s'\n", vendor.c_str());

    STRING_CLASS profile;
    status = device->getInfo(CL_DEVICE_PROFILE, &profile);
    printf("    profile: %s'\n", profile.c_str());

    STRING_CLASS version;
    status = device->getInfo(CL_DEVICE_VERSION, &version);
    printf("    version: %s'\n", version.c_str());

    STRING_CLASS extensions;
    status = device->getInfo(CL_DEVICE_EXTENSIONS, &extensions);
    printf("    extensions: %s'\n", extensions.c_str());

    // type of device
    status = device->getInfo(CL_DEVICE_TYPE, &val);
    if (status == CL_SUCCESS) {
        printf("    Type: ");
        if (val & CL_DEVICE_TYPE_DEFAULT) {
            val &= ~CL_DEVICE_TYPE_DEFAULT;
            printf("Default ");
        }
        if (val & CL_DEVICE_TYPE_CPU) {
            val &= ~CL_DEVICE_TYPE_CPU;
            printf("CPU ");
        }
        if (val & CL_DEVICE_TYPE_GPU) {
            val &= ~CL_DEVICE_TYPE_GPU;
            printf("GPU ");
        }
        if (val & CL_DEVICE_TYPE_ACCELERATOR) {
            val &= ~CL_DEVICE_TYPE_ACCELERATOR;
            printf("Accelerator ");
        }
        if (val != 0) {
            printf("Unknown (0x%llx)", val);
        }
        printf("\n");
    } else {
        printf("Unable to get TYPE: %s!\n", OpenclHelper::cl_error_to_str(status));
    }

    // device capabilities
    status = device->getInfo(CL_DEVICE_EXECUTION_CAPABILITIES, &val);
    if (status == CL_SUCCESS) {
        printf("    EXECUTION_CAPABILITIES: ");
        if (val & CL_EXEC_KERNEL) {
            val &= ~CL_EXEC_KERNEL;
            printf("Kernel ");
        }
        if (val & CL_EXEC_NATIVE_KERNEL) {
            val &= ~CL_EXEC_NATIVE_KERNEL;
            printf("Native ");
        }
        if (val) printf("Unknown (0x%llx) ", val);
        printf("\n");
    } else {
        printf("Unable to get EXECUTION_CAPABILITIES: %s!\n", OpenclHelper::cl_error_to_str(status));
    }

    status = device->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, &val);
    if (status == CL_SUCCESS) {
        printf("    GLOBAL_MEM_CACHE_TYPE: ");
        if (val == 0) {
            printf("None");
        } else if (val == 1) {
            printf("Read-Only");
        } else if (val == 2) {
            printf("Read-Write");
        } else {
            printf("??? %lld", val);
        }
        printf("\n");
    } else {
        printf("Unable to get GLOBAL_MEM_CACHE_TYPE: %s!\n", OpenclHelper::cl_error_to_str(status));
    }

    status = device->getInfo(CL_DEVICE_LOCAL_MEM_TYPE, &val);
    if (status == CL_SUCCESS) {
        printf("    DEVICE_LOCAL_MEM_TYPE: ");
        if (val == 1) {
            printf("Local");
        } else if (val == 2) {
            printf("Global");
        } else {
            printf("??? %lld", val);
        }
        printf("\n");
    } else {
        printf("Unable to get DEVICE_LOCAL_MEM_TYPE: %s!\n", OpenclHelper::cl_error_to_str(status));
    }

    VECTOR_CLASS<int64_t> size;
    status = device->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &size);
    if (status == CL_SUCCESS) {
        printf("    MAX_WORK_GROUP_SIZES: ");
        for (uint i = 0; i < size.size(); i++) {
            printf("%zd ", size[i]);
        }
        printf("\n");
    } else {
        printf("%s\n", OpenclHelper::cl_error_to_str(status));
    }

    map<int, const char*> longProps;
    longProps[CL_DEVICE_VENDOR_ID] = "VENDOR_ID";
    longProps[CL_DEVICE_MAX_COMPUTE_UNITS] = "MAX_COMPUTE_UNITS";
    longProps[CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS] = "MAX_WORK_ITEM_DIMENSIONS";
    longProps[CL_DEVICE_MAX_WORK_GROUP_SIZE] = "MAX_WORK_GROUP_SIZE";
    longProps[CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR] = "PREFERRED_VECTOR_WIDTH_CHAR";
    longProps[CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT] = "PREFERRED_VECTOR_WIDTH_SHORT";
    longProps[CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT] = "PREFERRED_VECTOR_WIDTH_INT";
    longProps[CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG] = "PREFERRED_VECTOR_WIDTH_LONG";
    longProps[CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT] = "PREFERRED_VECTOR_WIDTH_FLOAT";
    longProps[CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE] = "PREFERRED_VECTOR_WIDTH_DOUBLE";
    longProps[CL_DEVICE_MAX_CLOCK_FREQUENCY] = "MAX_CLOCK_FREQUENCY";
    longProps[CL_DEVICE_ADDRESS_BITS] = "ADDRESS_BITS";
    longProps[CL_DEVICE_MAX_MEM_ALLOC_SIZE] = "MAX_MEM_ALLOC_SIZE";
    longProps[CL_DEVICE_IMAGE_SUPPORT] = "IMAGE_SUPPORT";
    longProps[CL_DEVICE_MAX_READ_IMAGE_ARGS] = "MAX_READ_IMAGE_ARGS";
    longProps[CL_DEVICE_MAX_WRITE_IMAGE_ARGS] = "MAX_WRITE_IMAGE_ARGS";
    longProps[CL_DEVICE_IMAGE2D_MAX_WIDTH] = "IMAGE2D_MAX_WIDTH";
    longProps[CL_DEVICE_IMAGE2D_MAX_HEIGHT] = "IMAGE2D_MAX_HEIGHT";
    longProps[CL_DEVICE_IMAGE3D_MAX_WIDTH] = "IMAGE3D_MAX_WIDTH";
    longProps[CL_DEVICE_IMAGE3D_MAX_HEIGHT] = "IMAGE3D_MAX_HEIGHT";
    longProps[CL_DEVICE_IMAGE3D_MAX_DEPTH] = "IMAGE3D_MAX_DEPTH";
    longProps[CL_DEVICE_MAX_SAMPLERS] = "MAX_SAMPLERS";
    longProps[CL_DEVICE_MAX_PARAMETER_SIZE] = "MAX_PARAMETER_SIZE";
    longProps[CL_DEVICE_MEM_BASE_ADDR_ALIGN] = "MEM_BASE_ADDR_ALIGN";
    longProps[CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE] = "MIN_DATA_TYPE_ALIGN_SIZE";
    longProps[CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE] = "GLOBAL_MEM_CACHELINE_SIZE";
    longProps[CL_DEVICE_GLOBAL_MEM_CACHE_SIZE] = "GLOBAL_MEM_CACHE_SIZE";
    longProps[CL_DEVICE_GLOBAL_MEM_SIZE] = "GLOBAL_MEM_SIZE";
    longProps[CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE] = "MAX_CONSTANT_BUFFER_SIZE";
    longProps[CL_DEVICE_MAX_CONSTANT_ARGS] = "MAX_CONSTANT_ARGS";
    longProps[CL_DEVICE_LOCAL_MEM_SIZE] = "LOCAL_MEM_SIZE";
    longProps[CL_DEVICE_ERROR_CORRECTION_SUPPORT] = "ERROR_CORRECTION_SUPPORT";
    longProps[CL_DEVICE_PROFILING_TIMER_RESOLUTION] = "PROFILING_TIMER_RESOLUTION";
    longProps[CL_DEVICE_ENDIAN_LITTLE] = "ENDIAN_LITTLE";
    longProps[CL_DEVICE_AVAILABLE] = "AVAILABLE";
    longProps[CL_DEVICE_COMPILER_AVAILABLE] = "COMPILER_AVAILABLE";

    for (map<int, const char*>::iterator it=longProps.begin(); it!=longProps.end(); ++it) {
        status = device->getInfo(it->first, &val);
        if (status == CL_SUCCESS) {
            printf("    %s: %lld", it->second, val);
            printf("\n");
        } else {
            printf("%s\n", OpenclHelper::cl_error_to_str(status));
        }
    }
}


/**
 * Prints information about all devices from all platforms.
 * If param full is false, prints only names of devices.
 */
void OpenclHelper::print_platforms_devices(bool full) {
    VECTOR_CLASS<Platform> platforms;
    Platform::get(&platforms);

    for (uint i = 0; i < platforms.size(); i++) {
        STRING_CLASS name, version;
        platforms[i].getInfo(CL_PLATFORM_NAME, &name);
        platforms[i].getInfo(CL_PLATFORM_VERSION, &version);
        printf("platform %d: vendor '%s' - '%s'\n", i, name.c_str(), version.c_str());

        VECTOR_CLASS<Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (uint j = 0; j < devices.size(); j++) {
            if (full) {
                printf("  device %d:\n", j);
                print_device_info(&devices[j]);
            } else {
                STRING_CLASS dname;
                devices[j].getInfo(CL_DEVICE_NAME, &dname);
                printf("  device %d: '%s'\n", j, dname.c_str());
            }
        }
    }
}

/**
 * Returns device from specific platform.
 */
Device OpenclHelper::get_device(uint platformIndex, uint deviceIndex) {
    VECTOR_CLASS<Platform> platforms;
    Platform::get(&platforms);
    if (platformIndex + 1 > platforms.size()) {
        printf("Invalid platform index: %d\n", platformIndex);
        return NULL;
    }

    VECTOR_CLASS<Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (deviceIndex + 1 > devices.size()) {
        printf("Invalid device index: %d\n", deviceIndex);
        return NULL;
    }
    
    return devices[deviceIndex];
}

/**
 * Reads data from given filename.
 */
char *OpenclHelper::read_file(const char *filename)
{
    FILE *f = fopen(filename, "r");
    CHECK_SYS_ERROR(!f, (std::string("read_file: opening file: ") + filename).c_str());
  
    // figure out file size
    CHECK_SYS_ERROR(fseek(f, 0, SEEK_END) < 0, (std::string("read_file: seeking to end") + filename).c_str());
    std::size_t size = ftell(f);
  
    CHECK_SYS_ERROR(fseek(f, 0, SEEK_SET) != 0,
        (std::string("read_file: seeking to start") + filename).c_str());
  
    // allocate memory, slurp in entire file
    char *result = (char *) malloc(size+1);
    CHECK_SYS_ERROR(!result, (std::string("read_file: allocating file contents") + filename).c_str());
    CHECK_SYS_ERROR(fread(result, 1, size, f) < size,
        (std::string("read_file: reading file contents") + filename).c_str());
  
    // close, return
    CHECK_SYS_ERROR(fclose(f), (std::string("read_file: closing file") + filename).c_str());
    result[size] = '\0';
  
    return result;
}

/**
 * Builds program for all devices in context.
 */
Program OpenclHelper::build_program(Context ctx, std::vector<std::string> codes, char const *options)
{
    if (options && strlen(options) == 0)
    {
      // reportedly, some implementations dislike empty strings.
      options = NULL;
    }
    Program::Sources sources;
    for (uint i = 0; i < codes.size(); i++) {
        sources.push_back(std::make_pair(codes[i].c_str(), codes[i].length()));
    }
    cl_int status;
    Program program = Program(ctx, sources, &status);
    CHECK_CL_ERROR(status, "cl::Program");
  
    VECTOR_CLASS<Device> devices;
    program.getInfo(CL_PROGRAM_DEVICES, &devices);
    // build it
    status = program.build(devices, options);

    bool first_record = true;
    for (uint i = 0; i < devices.size(); i++) {
        STRING_CLASS log;
        // get build log and print it
        program.getBuildInfo(devices[i], CL_PROGRAM_BUILD_LOG, &log);
        if (log.length() > 1) {
            STRING_CLASS name;
            status = devices[i].getInfo(CL_DEVICE_NAME, &name);
            if (first_record){
                fprintf(stderr, "*** build of OpenCL program on "); 
            }
            fprintf(stderr, "%s said: %s\n", name.c_str(), log.c_str());
        }
    }
    CHECK_CL_ERROR(status, "cl::Program::build()");
  
    return program;
  }