// src/device/kernels/vector_kernels.cu
#include "../../../include/device/kernels/vector_kernels.cuh"
#include "../../../include/device/cuda_errors.cuh"

namespace cato {
namespace kernels {

/**
 * @brief Element-wise vector multiplication kernel
 * 
 * Performs element-wise multiplication of two vectors: result[i] = a[i] * b[i]
 * for all valid indices i < n.
 * 
 * @param a First input vector (device memory)
 * @param b Second input vector (device memory)
 * @param result Output vector (device memory)
 * @param n Number of elements in the vectors
 * 
 * @note This kernel is designed for vectors of double precision floating-point values.
 * @note The kernel uses a 1D grid configuration where each thread processes one element.
 * @note Input vectors must be of the same size and properly allocated on device memory.
 * 
 * @see launch_vecmul_kernel for the host function that launches this kernel
 */
__global__ void vecmul_kernel(const double* a, const double* b, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

/**
 * @brief Element-wise vector addition kernel
 * 
 * Performs element-wise addition of two vectors: result[i] = a[i] + b[i]
 * for all valid indices i < n.
 * 
 * @param a First input vector (device memory)
 * @param b Second input vector (device memory)
 * @param result Output vector (device memory)
 * @param n Number of elements in the vectors
 * 
 * @note This kernel is designed for vectors of double precision floating-point values.
 * @note The kernel uses a 1D grid configuration where each thread processes one element.
 * @note Input vectors must be of the same size and properly allocated on device memory.
 * 
 * @see launch_vecadd_kernel for the host function that launches this kernel
 */
__global__ void vecadd_kernel(const double* a, const double* b, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief Element-wise vector subtraction kernel
 * 
 * Performs element-wise subtraction of two vectors: result[i] = a[i] - b[i]
 * for all valid indices i < n.
 * 
 * @param a First input vector (device memory)
 * @param b Second input vector (device memory)
 * @param result Output vector (device memory)
 * @param n Number of elements in the vectors
 * 
 * @note This kernel is designed for vectors of double precision floating-point values.
 * @note The kernel uses a 1D grid configuration where each thread processes one element.
 * @note Input vectors must be of the same size and properly allocated on device memory.
 * 
 * @see launch_vecsub_kernel for the host function that launches this kernel
 */
__global__ void vecsub_kernel(const double* a, const double* b, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

/**
 * @brief Vector-scalar multiplication kernel
 * 
 * Performs scalar multiplication on a vector: result[i] = a[i] * scalar
 * for all valid indices i < n.
 * 
 * @param a Input vector (device memory)
 * @param scalar Scalar value to multiply with each element
 * @param result Output vector (device memory)
 * @param n Number of elements in the vector
 * 
 * @note This kernel is designed for vectors of double precision floating-point values.
 * @note The kernel uses a 1D grid configuration where each thread processes one element.
 * @note The scalar value is broadcast to all elements of the vector.
 * 
 * @see launch_vecmul_scalar_kernel for the host function that launches this kernel
 */
__global__ void vecmul_scalar_kernel(const double* a, double scalar, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * scalar;
    }
}

/**
 * @brief Vector-scalar addition kernel
 * 
 * Performs scalar addition on a vector: result[i] = a[i] + scalar
 * for all valid indices i < n.
 * 
 * @param a Input vector (device memory)
 * @param scalar Scalar value to add to each element
 * @param result Output vector (device memory)
 * @param n Number of elements in the vector
 * 
 * @note This kernel is designed for vectors of double precision floating-point values.
 * @note The kernel uses a 1D grid configuration where each thread processes one element.
 * @note The scalar value is broadcast to all elements of the vector.
 * 
 * @see launch_vecadd_scalar_kernel for the host function that launches this kernel
 */
__global__ void vecadd_scalar_kernel(const double* a, double scalar, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + scalar;
    }
}

/**
 * @brief Launch vector multiplication kernel with error checking
 * 
 * Host function that launches the vecmul_kernel with the specified grid and block
 * dimensions, and performs CUDA error checking after kernel execution.
 * 
 * @param a First input vector (device memory)
 * @param b Second input vector (device memory)
 * @param result Output vector (device memory)
 * @param n Number of elements in the vectors
 * @param grid Grid dimensions for kernel launch
 * @param block Block dimensions for kernel launch
 * @param stream CUDA stream for asynchronous execution (0 for default stream)
 * 
 * @note This function performs error checking using CHECK_CUDA_ERROR macro.
 * @note All input vectors must be properly allocated on device memory.
 * @note Grid and block dimensions should be calculated based on vector size n.
 * 
 * @see vecmul_kernel for the actual kernel implementation
 */
void launch_vecmul_kernel(const double* a, const double* b, double* result, 
                         size_t n, dim3 grid, dim3 block, cudaStream_t stream) {
    vecmul_kernel<<<grid, block, 0, stream>>>(a, b, result, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

/**
 * @brief Launch vector addition kernel with error checking
 * 
 * Host function that launches the vecadd_kernel with the specified grid and block
 * dimensions, and performs CUDA error checking after kernel execution.
 * 
 * @param a First input vector (device memory)
 * @param b Second input vector (device memory)
 * @param result Output vector (device memory)
 * @param n Number of elements in the vectors
 * @param grid Grid dimensions for kernel launch
 * @param block Block dimensions for kernel launch
 * @param stream CUDA stream for asynchronous execution (0 for default stream)
 * 
 * @note This function performs error checking using CHECK_CUDA_ERROR macro.
 * @note All input vectors must be properly allocated on device memory.
 * @note Grid and block dimensions should be calculated based on vector size n.
 * 
 * @see vecadd_kernel for the actual kernel implementation
 */
void launch_vecadd_kernel(const double* a, const double* b, double* result, 
                         size_t n, dim3 grid, dim3 block, cudaStream_t stream) {
    vecadd_kernel<<<grid, block, 0, stream>>>(a, b, result, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

/**
 * @brief Launch vector subtraction kernel with error checking
 * 
 * Host function that launches the vecsub_kernel with the specified grid and block
 * dimensions, and performs CUDA error checking after kernel execution.
 * 
 * @param a First input vector (device memory)
 * @param b Second input vector (device memory)
 * @param result Output vector (device memory)
 * @param n Number of elements in the vectors
 * @param grid Grid dimensions for kernel launch
 * @param block Block dimensions for kernel launch
 * @param stream CUDA stream for asynchronous execution (0 for default stream)
 * 
 * @note This function performs error checking using CHECK_CUDA_ERROR macro.
 * @note All input vectors must be properly allocated on device memory.
 * @note Grid and block dimensions should be calculated based on vector size n.
 * 
 * @see vecsub_kernel for the actual kernel implementation
 */
void launch_vecsub_kernel(const double* a, const double* b, double* result, 
                         size_t n, dim3 grid, dim3 block, cudaStream_t stream) {
    vecsub_kernel<<<grid, block, 0, stream>>>(a, b, result, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

} // namespace kernels
} // namespace cato
