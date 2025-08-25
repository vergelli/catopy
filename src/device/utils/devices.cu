// src/device/utils/devices.cu
#include "devices.cuh"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <stdexcept>

// Implementación de la clase Device
Device::Device(int device_id, const cudaDeviceProp& props) 
    : device_id_(device_id), properties_(props) {}

int Device::get_id() const { return device_id_; }
std::string Device::get_name() const { return properties_.name; }
std::string Device::get_compute_capability() const { 
    return std::to_string(properties_.major) + "." + std::to_string(properties_.minor); 
}
size_t Device::get_total_global_memory() const { return properties_.totalGlobalMem; }
size_t Device::get_shared_memory_per_block() const { return properties_.sharedMemPerBlock; }
int Device::get_registers_per_block() const { return properties_.regsPerBlock; }
int Device::get_warp_size() const { return properties_.warpSize; }
int Device::get_max_threads_per_block() const { return properties_.maxThreadsPerBlock; }
int Device::get_max_threads_per_multiprocessor() const { return properties_.maxThreadsPerMultiProcessor; }
int Device::get_multiprocessor_count() const { return properties_.multiProcessorCount; }
int Device::get_memory_bus_width() const { return properties_.memoryBusWidth; }
int Device::get_l2_cache_size() const { return properties_.l2CacheSize; }
size_t Device::get_total_constant_memory() const { return properties_.totalConstMem; }

std::tuple<int, int, int> Device::get_max_grid_size() const {
    return std::make_tuple(properties_.maxGridSize[0], properties_.maxGridSize[1], properties_.maxGridSize[2]);
}

std::tuple<int, int, int> Device::get_max_threads_dim() const {
    return std::make_tuple(properties_.maxThreadsDim[0], properties_.maxThreadsDim[1], properties_.maxThreadsDim[2]);
}

void Device::show() const {
    printf("Device ID: %d, Name: %s, SMs: %d\n", 
           device_id_, properties_.name, properties_.multiProcessorCount);
}

std::unordered_map<std::string, std::string> Device::get_properties() const {
    std::unordered_map<std::string, std::string> device_info;
    device_info["Device ID"] = std::to_string(device_id_);
    device_info["Name"] = properties_.name;
    device_info["Compute Capability"] = get_compute_capability();
    device_info["Total Global Memory"] = std::to_string(properties_.totalGlobalMem);
    device_info["Shared Memory Per Block"] = std::to_string(properties_.sharedMemPerBlock);
    device_info["Registers Per Block"] = std::to_string(properties_.regsPerBlock);
    device_info["Warp Size"] = std::to_string(properties_.warpSize);
    device_info["Max Threads Per Block"] = std::to_string(properties_.maxThreadsPerBlock);
    device_info["Max Threads Per MultiProcessor"] = std::to_string(properties_.maxThreadsPerMultiProcessor);
    device_info["Number of SMs"] = std::to_string(properties_.multiProcessorCount);
    device_info["Memory Bus Width (bits)"] = std::to_string(properties_.memoryBusWidth);
    device_info["L2 Cache Size"] = std::to_string(properties_.l2CacheSize);
    
    // Convertir tuples a strings formateados
    auto grid_size = get_max_grid_size();
    device_info["Max Grid Size"] = "(" + std::to_string(std::get<0>(grid_size)) + ", " +
                                   std::to_string(std::get<1>(grid_size)) + ", " +
                                   std::to_string(std::get<2>(grid_size)) + ")";
    
    auto threads_dim = get_max_threads_dim();
    device_info["Max Threads Dim"] = "(" + std::to_string(std::get<0>(threads_dim)) + ", " +
                                     std::to_string(std::get<1>(threads_dim)) + ", " +
                                     std::to_string(std::get<2>(threads_dim)) + ")";
    
    device_info["Total Constant Memory"] = std::to_string(properties_.totalConstMem);
    return device_info;
}

// Implementación de la clase Devices
class Devices::Impl {
public:
    Impl() {
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);

        // Debugging: verificar si hay error en cudaGetDeviceCount
        if (error != cudaSuccess) {
            printf("ERROR: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error));
            device_count = 0;
        }

        // Validación: device_count debe ser razonable
        if (device_count < 0 || device_count > 100) {
            printf("WARNING: device_count seems invalid: %d, setting to 0\n", device_count);
            device_count = 0;
        }

        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp props;
            error = cudaGetDeviceProperties(&props, i);
            
            if (error != cudaSuccess) {
                printf("ERROR: cudaGetDeviceProperties(%d) failed: %s\n", i, cudaGetErrorString(error));
                continue;
            }
            
            // Validación: verificar que las propiedades sean razonables
            if (props.multiProcessorCount < 0 || props.multiProcessorCount > 1000) {
                printf("WARNING: Device %d has invalid SMs: %d\n", i, props.multiProcessorCount);
            }
            
            devices_.push_back(props);
        }
        
    }

    size_t count() const {
        return devices_.size();
    }

    Device get_device(int device_id) const {
        if (device_id < 0 || static_cast<size_t>(device_id) >= devices_.size()) {
            throw std::out_of_range("Device ID out of range: " + std::to_string(device_id));
        }
        return Device(device_id, devices_[device_id]);
    }

    void show() const {
        for (size_t i = 0; i < devices_.size(); ++i) {
            printf("Device ID: %zu, Name: %s, SMs: %d\n", 
                   i, devices_[i].name, devices_[i].multiProcessorCount);
        }
    }

private:
    std::vector<cudaDeviceProp> devices_;
};

Devices::Devices() : impl_(std::make_unique<Impl>()) {}
Devices::~Devices() = default;

size_t Devices::count() const {
    return impl_->count();
}

Device Devices::get_device(int device_id) const {
    return impl_->get_device(device_id);
}

void Devices::show() const {
    impl_->show();
}

Device Devices::operator[](int device_id) const {
    return get_device(device_id);
}



