#ifndef CATENSOR_H
#define CATENSOR_H

#include "device/GpuBuffer.cuh"
#include "device/caTensorInitZeroWrapper.cuh"
#include "device/caTensorInitValueWrapper.cuh"

#include <iostream>
#include <vector>
#include <stdexcept>

enum class Device {CPU,GPU};

class caTensor {
private:
    std::vector<size_t> shape;
    size_t total_size;
    Device device;

    std::vector<double> host_data;
    GpuBuffer<double> gpu_data;

public:

    caTensor(const std::vector<size_t>& shape, Device device = Device::CPU)
        : shape(shape),
        device(device),
        total_size(compute_size(shape)),
        host_data(device == Device::CPU ? total_size : 0, 0.0),
        gpu_data(device == Device::GPU ? total_size : 0) 
        {
            if (device == Device::GPU) {
                void init_tensor_zeros(double* gpu_ptr, size_t total_size);
            }
        }

    caTensor(const std::vector<size_t>& shape, double init_value, Device device = Device::CPU)
        : shape(shape),
        device(device),
        total_size(compute_size(shape)),
        host_data(device == Device::CPU ? std::vector<double>(compute_size(shape), init_value) : std::vector<double>()),
        gpu_data(device == Device::GPU ? compute_size(shape) : 0)
        {
            if (device == Device::GPU) {
                gpu_data = GpuBuffer<double>(total_size);
                init_tensor_by_value(gpu_data.get(), total_size, init_value);
            }
        }

    size_t size() const { return total_size; }

    double* cpu_ptr() {
        if (device != Device::CPU) throw std::runtime_error("Tensor is on GPU");
        return host_data.data();
    }

    double* gpu_ptr() {
        if (device != Device::GPU) throw std::runtime_error("Tensor is on CPU");
        return gpu_data.get();
    }

    const std::vector<size_t>& get_shape() const {
        return shape;
    }

private:
    static size_t compute_size(const std::vector<size_t>& shape) {
        size_t size = 1;
        for (auto dim : shape) size *= dim;
        return size;
    }
};

#endif






