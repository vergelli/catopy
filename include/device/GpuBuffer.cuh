

#ifndef GPU_BUFFER_H
#define GPU_BUFFER_H

#include "cuda_errors.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

template<typename T>
class GpuBuffer {
private:
    T* ptr;
    size_t size;

public:
    GpuBuffer(size_t count) : size(count) {
        CHECK_CUDA_ERROR(cudaMalloc((void**)&ptr, size * sizeof(T)));
    }

    ~GpuBuffer() {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }

    T* get() {
        return ptr;
    }

    size_t get_size() const {
        return size;
    }

    //TODO: Study hard this

    // Forbide copy
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // Allow move
    GpuBuffer(GpuBuffer&& other) noexcept : ptr(other.ptr), size(other.size) {
        other.ptr = nullptr;
    }

    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            cudaFree(ptr);
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
        }
        return *this;
    }
};

#endif
