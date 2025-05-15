// src/device/devices.cu
#include "devices.cuh"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>

class Devices::Impl {
public:
    Impl() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            devices_.push_back(props);
        }
    }

    std::unordered_map<std::string, std::string> get_properties(int device_id) const {
        const auto& d = devices_.at(device_id);
        std::unordered_map<std::string, std::string> device_info;
        device_info["Device ID"] = std::to_string(device_id);
        device_info["Name"] = d.name;
        device_info["Compute Capability"] = std::to_string(d.major) + "." + std::to_string(d.minor);
        device_info["Total Global Memory"] = std::to_string(d.totalGlobalMem);
        device_info["Shared Memory Per Block"] = std::to_string(d.sharedMemPerBlock);
        device_info["Registers Per Block"] = std::to_string(d.regsPerBlock);
        device_info["Warp Size"] = std::to_string(d.warpSize);
        device_info["Max Threads Per Block"] = std::to_string(d.maxThreadsPerBlock);
        device_info["Max Threads Per MultiProcessor"] = std::to_string(d.maxThreadsPerMultiProcessor);
        device_info["Number of SMs"] = std::to_string(d.multiProcessorCount);
        device_info["Clock Rate (KHz)"] = std::to_string(d.clockRate);
        device_info["Memory Clock Rate (KHz)"] = std::to_string(d.memoryClockRate);
        device_info["Memory Bus Width (bits)"] = std::to_string(d.memoryBusWidth);
        device_info["L2 Cache Size"] = std::to_string(d.l2CacheSize);
        device_info["Max Grid Size"] = "(" +
            std::to_string(d.maxGridSize[0]) + ", " +
            std::to_string(d.maxGridSize[1]) + ", " +
            std::to_string(d.maxGridSize[2]) + ")";
        device_info["Max Threads Dim"] = "(" +
            std::to_string(d.maxThreadsDim[0]) + ", " +
            std::to_string(d.maxThreadsDim[1]) + ", " +
            std::to_string(d.maxThreadsDim[2]) + ")";
        device_info["Total Constant Memory"] = std::to_string(d.totalConstMem);
        return device_info;
    }

    void print_devices() const {
        for (size_t i = 0; i < devices_.size(); ++i) {
            printf("Device ID: %zu, Name: %s, SMs: %d\n", i, devices_[i].name, devices_[i].multiProcessorCount);
        }
    }

    std::vector<std::unordered_map<std::string, std::string>> get_devices() const {
        std::vector<std::unordered_map<std::string, std::string>> device_list;
        for (size_t i = 0; i < devices_.size(); ++i) {
            device_list.push_back(get_properties(i));
        }
        return device_list;
    }

private:
    std::vector<cudaDeviceProp> devices_;
};

Devices::Devices() : impl_(std::make_unique<Impl>()) {}
Devices::~Devices() = default;

std::unordered_map<std::string, std::string> Devices::get_properties(int device_id) const {
    return impl_->get_properties(device_id);
}

void Devices::print_devices() const {
    impl_->print_devices();
}

std::vector<std::unordered_map<std::string, std::string>> Devices::get_devices() const {
    return impl_->get_devices();
}



