#ifndef MEMORY_TRANSFER_CUH
#define MEMORY_TRANSFER_CUH

#include "../cuda_errors.cuh"
#include <cuda_runtime.h>

/**
 * @file MemoryTransfer.cuh
 * @brief Namespace for HOST ↔ GPU memory transfer operations
 * * This namespace provides simple and efficient functions for transferring
 * data between HOST (CPU) memory and GPU memory.
 * * Features:
 * - Template functions for any data type
 * - Automatic CUDA error handling
 * - Easy to extend for future functionalities (CUDA streams, etc.)
 * * Usage:
 * @code
 * // Transfer data from HOST to GPU
 * MemoryTransfer::host_to_gpu(gpu_ptr, host_ptr, count);
 * * // Transfer data from GPU to HOST
 * MemoryTransfer::gpu_to_host(host_ptr, gpu_ptr, count);
 * @endcode
 */

namespace MemoryTransfer {

    /**
     * @brief Transfers data from HOST to GPU
     * @param gpu_ptr Pointer to GPU memory
     * @param host_ptr Pointer to HOST memory
     * @param count Number of elements to transfer
     * * This function is synchronous and blocks until the transfer is complete.
     */
    template<typename T>
    void host_to_gpu(T* gpu_ptr, const T* host_ptr, size_t count) {
        CHECK_CUDA_ERROR(cudaMemcpy(
            gpu_ptr, 
            host_ptr, 
            count * sizeof(T), 
            cudaMemcpyHostToDevice
        ));
    }

    /**
     * @brief Transfers data from GPU to HOST
     * @param host_ptr Pointer to HOST memory
     * @param gpu_ptr Pointer to GPU memory
     * @param count Number of elements to transfer
     * * This function is synchronous and blocks until the transfer is complete.
     */
    template<typename T>
    void gpu_to_host(T* host_ptr, const T* gpu_ptr, size_t count) {
        CHECK_CUDA_ERROR(cudaMemcpy(
            host_ptr, 
            gpu_ptr, 
            count * sizeof(T), 
            cudaMemcpyDeviceToHost
        ));
    }

    /**
     * @brief Transfers data within the GPU (device to device)
     * @param dst_ptr Destination pointer on GPU
     * @param src_ptr Source pointer on GPU
     * @param count Number of elements to transfer
     * * This function is useful for internal GPU operations.
     */
    template<typename T>
    void gpu_to_gpu(T* dst_ptr, const T* src_ptr, size_t count) {
        CHECK_CUDA_ERROR(cudaMemcpy(
            dst_ptr, 
            src_ptr, 
            count * sizeof(T), 
            cudaMemcpyDeviceToDevice
        ));
    }

    /**
     * @brief Transfers data within the HOST (host to host)
     * @param dst_ptr Destination pointer on HOST
     * @param src_ptr Source pointer on HOST
     * @param count Number of elements to transfer
     * * This function is useful for internal HOST operations.
     */
    template<typename T>
    void host_to_host(T* dst_ptr, const T* src_ptr, size_t count) {
        CHECK_CUDA_ERROR(cudaMemcpy(
            dst_ptr, 
            src_ptr, 
            count * sizeof(T), 
            cudaMemcpyHostToHost
        ));
    }

    /**
     * @brief Gets the transfer size in bytes
     * @param count Number of elements
     * @return Total size in bytes
     */
    template<typename T>
    size_t get_transfer_size(size_t count) {
        return count * sizeof(T);
    }

    /**
     * @brief Checks if pointers are valid before transfer
     * @param ptr Pointer to check
     * @param count Number of elements
     * @return true if the pointer is valid, false otherwise
     */
    template<typename T>
    bool is_valid_pointer(const T* ptr, size_t count) {
        return ptr != nullptr && count > 0;
    }

    // TODO: FUTURO - Transferencias asíncronas con CUDA streams 
    //   o alguna gilada similar NO teno idea pero podria esclar.

    //* template<typename T>
    //* void host_to_gpu_async(T* gpu_ptr, const T* host_ptr, size_t count, cudaStream_t stream) {
    //*     CHECK_CUDA_ERROR(cudaMemcpyAsync(gpu_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    //* }

    // TODO: FUTURO - Transferencias con memoria pinned para mejor rendimiento
    // Esto lo vi pero todavia tengo que estudiarlo un poco mas a fondo.

    // template<typename T>
    // void host_to_gpu_pinned(T* gpu_ptr, const T* host_ptr, size_t count);

} // namespace MemoryTransfer

#endif // MEMORY_TRANSFER_CUH
