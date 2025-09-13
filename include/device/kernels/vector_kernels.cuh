// include/device/kernels/vector_kernels.cuh
#ifndef VECTOR_KERNELS_CUH
#define VECTOR_KERNELS_CUH

#include <cuda_runtime.h>
#include <cstddef>

namespace cato {
namespace kernels {

/**
 * @brief Kernel para multiplicación element-wise de vectores
 * @param a Primer vector (entrada)
 * @param b Segundo vector (entrada)
 * @param result Vector resultado (salida)
 * @param n Número de elementos
 */
__global__ void vecmul_kernel(const double* a, const double* b, double* result, size_t n);

/**
 * @brief Kernel para suma element-wise de vectores
 * @param a Primer vector (entrada)
 * @param b Segundo vector (entrada)
 * @param result Vector resultado (salida)
 * @param n Número de elementos
 */
__global__ void vecadd_kernel(const double* a, const double* b, double* result, size_t n);

/**
 * @brief Kernel para resta element-wise de vectores
 * @param a Primer vector (entrada)
 * @param b Segundo vector (entrada)
 * @param result Vector resultado (salida)
 * @param n Número de elementos
 */
__global__ void vecsub_kernel(const double* a, const double* b, double* result, size_t n);

/**
 * @brief Kernel para multiplicación por escalar
 * @param a Vector (entrada)
 * @param scalar Escalar
 * @param result Vector resultado (salida)
 * @param n Número de elementos
 */
__global__ void vecmul_scalar_kernel(const double* a, double scalar, double* result, size_t n);

/**
 * @brief Kernel para suma con escalar
 * @param a Vector (entrada)
 * @param scalar Escalar
 * @param result Vector resultado (salida)
 * @param n Número de elementos
 */
__global__ void vecadd_scalar_kernel(const double* a, double scalar, double* result, size_t n);

void launch_vecmul_kernel(const double* a, const double* b, double* result, 
                         size_t n, dim3 grid, dim3 block, cudaStream_t stream = 0);

void launch_vecadd_kernel(const double* a, const double* b, double* result, 
                         size_t n, dim3 grid, dim3 block, cudaStream_t stream = 0);

void launch_vecsub_kernel(const double* a, const double* b, double* result, 
                         size_t n, dim3 grid, dim3 block, cudaStream_t stream = 0);

} // namespace kernels
} // namespace cato

#endif // VECTOR_KERNELS_CUH
