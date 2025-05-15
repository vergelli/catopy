#ifndef CUDA_ERRORS_CUH
#define CUDA_ERRORS_CUH

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(call)                                                \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in function " << __func__ << ": "        \
                      << cudaGetErrorString(err) << " at line " << __LINE__   \
                      << std::endl;                                           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#endif // CUDA_ERRORS_CUH
