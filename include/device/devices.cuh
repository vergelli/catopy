#ifndef DEVICES_CUH
#define DEVICES_CUH

#include <string>
#include <vector>
#include <memory> // For std::unique_ptr
#include <unordered_map> // Para std::unordered_map
#include <tuple> // Para std::tuple
#include <cuda_runtime.h>

// Forward declaration
class Device;

class Devices {
public:
    Devices();
    ~Devices(); // Necesario para PIMPL

    // Retorna el número total de dispositivos disponibles
    size_t count() const;

    // Retorna una instancia de Device para el ID especificado
    Device get_device(int device_id) const;

    // Método para mostrar información de todos los dispositivos
    void show() const;
    
    // Operador [] para acceder a dispositivos por ID
    Device operator[](int device_id) const;

private:
    class Impl; // Declaración opaca
    std::unique_ptr<Impl> impl_; // PIMPL
};

// Nueva clase para representar un dispositivo individual
class Device {
public:
    Device(int device_id, const cudaDeviceProp& props);

    // Getters para propiedades del dispositivo
    int get_id() const;
    std::string get_name() const;
    std::string get_compute_capability() const;
    size_t get_total_global_memory() const;
    size_t get_shared_memory_per_block() const;
    int get_registers_per_block() const;
    int get_warp_size() const;
    int get_max_threads_per_block() const;
    int get_max_threads_per_multiprocessor() const;
    int get_multiprocessor_count() const;
    int get_memory_bus_width() const;
    int get_l2_cache_size() const;
    std::tuple<int, int, int> get_max_grid_size() const;
    std::tuple<int, int, int> get_max_threads_dim() const;
    size_t get_total_constant_memory() const;

    // Método para mostrar información del dispositivo
    void show() const;

    // Método para obtener todas las propiedades como mapa
    std::unordered_map<std::string, std::string> get_properties() const;

private:
    int device_id_;
    cudaDeviceProp properties_;
};

#endif