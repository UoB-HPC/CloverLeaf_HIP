#if defined(MPI_HDR)
extern "C" void clover_abort_();
#endif

#include "cuda_common.hpp"
#include <cstdarg>
#include <cstdio>

static const char* errorCodes
(int err_code)
{
    switch(err_code)
    {
        case hipSuccess: return "hipSuccess"; // 0
        case hipErrorMissingConfiguration: return "hipErrorMissingConfiguration"; // 1
        case hipErrorOutOfMemory: return "hipErrorOutOfMemory"; // 2
        case hipErrorNotInitialized: return "hipErrorNotInitialized"; // 3
        case hipErrorLaunchFailure: return "hipErrorLaunchFailure"; // 4
        case hipErrorPriorLaunchFailure: return "hipErrorPriorLaunchFailure"; // 5
        case hipErrorLaunchTimeOut: return "hipErrorLaunchTimeOut"; // 6
        case hipErrorLaunchOutOfResources: return "hipErrorLaunchOutOfResources"; // 7
        case hipErrorInvalidDeviceFunction: return "hipErrorInvalidDeviceFunction"; // 8
        case hipErrorInvalidConfiguration: return "hipErrorInvalidConfiguration"; // 9
        case hipErrorInvalidDevice: return "hipErrorInvalidDevice"; // 10
        case hipErrorInvalidValue: return "hipErrorInvalidValue";// 11
        case hipErrorInvalidPitchValue: return "hipErrorInvalidPitchValue";// 12
        case hipErrorInvalidSymbol: return "hipErrorInvalidSymbol";// 13
        case hipErrorMapFailed: return "hipErrorMapFailed";// 14
        case hipErrorUnmapFailed: return "hipErrorUnmapFailed";// 15
        // case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";// 16
        case hipErrorInvalidDevicePointer: return "hipErrorInvalidDevicePointer";// 17
        // case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";// 18
        // case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";// 19
        // case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";// 20
        case hipErrorInvalidMemcpyDirection: return "hipErrorInvalidMemcpyDirection";// 21
        // case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";// 22
        // case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";// 23
        // case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";// 24
        // case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";// 25
        // case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";// 26
        // case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";// 27
        // case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";// 28
        case hipErrorDeinitialized: return "hipErrorDeinitialized";// 29
        case hipErrorUnknown: return "hipErrorUnknown";// 30
        // case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";// 31
        // case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";// 32
        case hipErrorInvalidHandle: return "hipErrorInvalidHandle";// 33
        case hipErrorNotReady: return "hipErrorNotReady";// 34
        case hipErrorInsufficientDriver: return "hipErrorInsufficientDriver";// 35
        case hipErrorSetOnActiveProcess: return "hipErrorSetOnActiveProcess";// 36
        // case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface";// 37
        case hipErrorNoDevice: return "hipErrorNoDevice";// 38
        case hipErrorECCNotCorrectable: return "hipErrorECCNotCorrectable";// 39
        case hipErrorSharedObjectSymbolNotFound: return "hipErrorSharedObjectSymbolNotFound";// 40
        case hipErrorSharedObjectInitFailed: return "hipErrorSharedObjectInitFailed";// 41
        case hipErrorUnsupportedLimit: return "hipErrorUnsupportedLimit";// 42
        // case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName";// 43
        // case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName";// 44
        // case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName";// 45
        // case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable";// 46
        case hipErrorInvalidImage: return "hipErrorInvalidImage";// 47
        case hipErrorNoBinaryForGpu: return "hipErrorNoBinaryForGpu";// 48
        // case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext";// 49
        case hipErrorPeerAccessAlreadyEnabled: return "hipErrorPeerAccessAlreadyEnabled";// 50
        case hipErrorPeerAccessNotEnabled: return "hipErrorPeerAccessNotEnabled";// 51
        case hipErrorContextAlreadyInUse: return "hipErrorContextAlreadyInUse";// 52
        case hipErrorProfilerDisabled: return "hipErrorProfilerDisabled";// 53
        case hipErrorProfilerNotInitialized: return "hipErrorProfilerNotInitialized";// 54
        case hipErrorProfilerAlreadyStarted: return "hipErrorProfilerAlreadyStarted";// 55
        case hipErrorProfilerAlreadyStopped: return "hipErrorProfilerAlreadyStopped";// 56
        case hipErrorAssert: return "hipErrorAssert";// 57
        // case cudaErrorTooManyPeers: return "cudaErrorTooManyPeers";// 58
        case hipErrorHostMemoryAlreadyRegistered: return "hipErrorHostMemoryAlreadyRegistered";// 59
        case hipErrorHostMemoryNotRegistered: return "hipErrorHostMemoryNotRegistered";// 60
        case hipErrorOperatingSystem: return "hipErrorOperatingSystem";// 61
        // case cudaErrorStartupFailure: return "cudaErrorStartupFailure";// 62
        // case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";// 63
        default: return "Unknown error";
    }
}

void CloverleafCudaChunk::errorHandler
(int line_num, const char* file)
{
    hipDeviceSynchronize();
    int l_e = hipGetLastError();
    if (hipSuccess != l_e)
    {
        cloverDie(line_num, file, "Error in %s - return code %d (%s)\n", file, l_e, errorCodes(l_e));
    }
}

// print out timing info when done
CloverleafCudaChunk::~CloverleafCudaChunk
(void)
{
    if (profiler_on)
    {
        fprintf(stdout, "@@@@@ PROFILING @@@@@\n");

        for (std::map<std::string, double>::iterator ii = kernel_times.begin();
            ii != kernel_times.end(); ii++)
        {
            fprintf(stdout, "%35s : %.3f\n", ii->first.c_str(), ii->second);
        }
    }
}

std::vector<double> CloverleafCudaChunk::dumpArray
(const std::string& arr_name, int x_extra, int y_extra)
{
    std::vector<double> host_arr(BUFSZ2D(x_extra, y_extra)/sizeof(double));

    hipDeviceSynchronize();

    try
    {
        hipMemcpy(&host_arr.front(), arr_names.at(arr_name),
            BUFSZ2D(x_extra, y_extra), hipMemcpyDeviceToHost);
    }
    catch (std::out_of_range e)
    {
        DIE("Error - %s was not in the arr_names map\n", arr_name.c_str());
    }

    errorHandler(__LINE__, __FILE__);

    return host_arr;
}

// called when something goes wrong
void CloverleafCudaChunk::cloverDie
(int line, const char* filename, const char* format, ...)
{
    fprintf(stderr, "@@@@@\n");
    fprintf(stderr, "\x1b[31m");
    fprintf(stderr, "Fatal error at line %d in %s:", line, filename);
    fprintf(stderr, "\x1b[0m");
    fprintf(stderr, "\n");

    va_list arglist;
    va_start(arglist, format);
    vfprintf(stderr, format, arglist);
    va_end(arglist);

    // TODO add logging or something

    fprintf(stderr, "\nExiting\n");

#if defined(MPI_HDR)
    clover_abort_();
#else
    exit(1);
#endif
}

