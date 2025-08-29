#ifndef GPU_BUFFER_CUH
#define GPU_BUFFER_CUH

/**
 * @file GpuBuffer.cuh
 * @brief Class for automatic and safe GPU memory management
 * * This class implements the RAII (Resource Acquisition Is Initialization) pattern
 * to handle GPU memory automatically and safely.
 * * Key Features:
 * - Automatic allocation of GPU memory upon object creation
 * - Automatic deallocation of memory upon object destruction
 * - Prevention of memory leaks and double-frees
 * - Support for move semantics (transfer of ownership)
 * - Prohibition of accidental copies
 */

#include "../cuda_errors.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include "../../logger/Logger.cuh"

template<typename T>
class GpuBuffer {
private:
    T* gpu_pointer = nullptr;        // Pointer to GPU memory
    size_t element_count = 0;        // Number of stored elements

public:
    /**
     * @brief Constructor: automatically allocates GPU memory
     * @param count Number of elements of type T to store
     * * Example:
     * @code
     * GpuBuffer<double> buffer(1000); // Stores 1000 doubles on the GPU
     * @endcode
     */
    GpuBuffer(size_t count) : element_count(count) {
        Logger::debug("GpuBuffer constructor called with count= {}", count);
        Logger::debug("sizeof(T) =  {}", sizeof(T));
        Logger::debug("Total bytes to allocate =  {}", (count * sizeof(T)));
        
        if (count == 0) {
            Logger::debug("ERROR - count is 0");
            throw std::invalid_argument("GpuBuffer: count must be greater than 0");
        }

        Logger::debug("About to call cudaMalloc...");

        // Allocate memory on GPU using CUDA
        try {
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_pointer, count * sizeof(T)));
            Logger::debug("cudaMalloc completed successfully");
        } catch (const std::exception& e) {
            Logger::debug("ERROR - cudaMalloc failed:  {}", e.what());
            throw;
        }

        Logger::debug("gpu_pointer =  {}", static_cast<void*>(gpu_pointer));
        
        if (gpu_pointer == nullptr) {
            Logger::debug("ERROR - gpu_pointer is null after cudaMalloc");
            throw std::runtime_error("GpuBuffer: Failed to allocate GPU memory");
        }
        
        Logger::debug("GpuBuffer constructor completed successfully");
    }

    /**
     * @brief Destructor: automatically deallocates GPU memory
     * * You don't need to call this manually. It automatically executes
     * when the object goes out of scope.
     */
    ~GpuBuffer() {
        Logger::debug("GpuBuffer destructor called");
        Logger::debug("gpu_pointer =  {}", static_cast<void*>(gpu_pointer));
        Logger::debug("element_count =  {}", element_count);
        
        if (gpu_pointer != nullptr) {
            Logger::debug("About to call cudaFree on gpu_pointer =  {}", static_cast<void*>(gpu_pointer));
            try {
                CHECK_CUDA_ERROR(cudaFree(gpu_pointer));
                Logger::debug("cudaFree completed successfully");
            } catch (const std::exception& e) {
                Logger::debug("ERROR - cudaFree failed:  {}", e.what());
            } catch (...) {
                Logger::debug("ERROR - Unknown exception in cudaFree");
            }
            gpu_pointer = nullptr;
        } else {
            Logger::debug("gpu_pointer was already nullptr in destructor");
        }
        
        Logger::debug("About to exit GpuBuffer destructor");
        Logger::debug("GpuBuffer destructor completed");
    }

    /**
     * @brief Get pointer to GPU memory
     * @return Raw pointer to the GPU memory
     *
     * Use this pointer for CUDA operations.
     * DO NOT manually free it - GpuBuffer handles that.
     */
    T* get_pointer() const { 
        return gpu_pointer; 
    }

    /**
     * @brief Get number of stored elements
     * @return Number of elements of type T
     */
    size_t get_size() const { 
        return element_count; 
    }

    /**
     * @brief Get total size in bytes
     * @return Total memory size in bytes
     */
    size_t get_size_bytes() const { 
        return element_count * sizeof(T); 
    }

    /**
     * @brief Check if the buffer is empty
     * @return true if there are no elements, false otherwise
     */
    bool is_empty() const { 
        return element_count == 0; 
    }

    // ===== PROHIBITION OF COPIES =====
    // Do not allow accidental copies that could cause double-free

    /**
     * @brief Copy Constructor FORBIDDEN
     *
     * GPU buffers cannot be copied because each one must have
     * its own memory. Copying could cause a double-free.
     */
    GpuBuffer(const GpuBuffer&) = delete;

    /**
     * @brief Copy Assignment Operator FORBIDDEN
     *
     * Same reason: do not allow accidental copies.
     */
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // ===== SUPPORT FOR MOVE SEMANTICS =====
    // Allow transferring ownership from one buffer to another

    /**
     * @brief Move Constructor: transfers ownership
     * @param other Buffer from which to take ownership
     * * After the move, 'other' is left in an invalid state
     * and should not be used.
     */
    GpuBuffer(GpuBuffer&& other) noexcept
        : gpu_pointer(other.gpu_pointer), element_count(other.element_count) {
        // Transfer ownership
        other.gpu_pointer = nullptr;
        other.element_count = 0;
    }

    /**
     * @brief Move Assignment Operator: transfers ownership
     * @param other Buffer from which to take ownership
     * @return Reference to this buffer
     *
     * Frees the current memory before taking the new one.
     */
    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            // Free current memory
            if (gpu_pointer != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(gpu_pointer));
            }

            // Transfer ownership
            gpu_pointer = other.gpu_pointer;
            element_count = other.element_count;

            // Invalidate 'other'
            other.gpu_pointer = nullptr;
            other.element_count = 0;
        }
        return *this;
    }
};

#endif // GPU_BUFFER_CUH