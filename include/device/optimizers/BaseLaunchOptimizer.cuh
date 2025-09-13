// include/device/optimizers/BaseLaunchOptimizer.cuh
#ifndef BASE_LAUNCH_OPTIMIZER_CUH
#define BASE_LAUNCH_OPTIMIZER_CUH

#include "../KernelOptimizerConstants.cuh"
#include "../devices.cuh"
#include "../LaunchConfig.cuh"
#include "../DataStructureInfo.cuh"
#include "../../logger/Logger.cuh"
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace cato {

// Forward declarations (ya incluidas en KernelLaunchOptimizer.cuh)
// class DataStructureInfo; - ya incluido
// class VectorInfo; - ya incluido
// class MatrixInfo; - ya incluido
// class Tensor3DInfo; - ya incluido
// class TensorNDInfo; - ya incluido
// struct LaunchConfig; - ya incluido

/**
 * @brief Clase base abstracta para optimizadores de lanzamiento de kernels
 * 
 * Esta clase proporciona la interfaz común y funcionalidad compartida
 * para todos los optimizadores especializados (Vector, Matrix, Tensor).
 */
class BaseLaunchOptimizer {
protected:
    Device device_;
    std::string optimization_strategy_;
    double occupancy_weight_;
    double bandwidth_weight_;
    
    // Constructor protegido para clases derivadas
    BaseLaunchOptimizer(const Device& device, 
                       const std::string& strategy = KernelOptimizerConstants::STRATEGY_BALANCED,
                       double occupancy_weight = KernelOptimizerConstants::DEFAULT_OCCUPANCY_WEIGHT,
                       double bandwidth_weight = KernelOptimizerConstants::DEFAULT_BANDWIDTH_WEIGHT);

public:
    virtual ~BaseLaunchOptimizer() = default;
    
    // Métodos virtuales puros que deben implementar las clases derivadas
    virtual LaunchConfig optimize(const DataStructureInfo& data_info) = 0;
    virtual double analyze_occupancy(int block_size) const = 0;
    virtual double estimate_bandwidth_utilization(const DataStructureInfo& data_info) const = 0;
    
    // Métodos comunes para todas las clases derivadas
    int get_optimal_warp_size() const;
    int calculate_blocks_needed(int block_size, size_t total_elements) const;
    bool validate_launch_config(const LaunchConfig& config) const;
    
    // Getters
    const Device& get_device() const { return device_; }
    std::string get_optimization_strategy() const { return optimization_strategy_; }
    double get_occupancy_weight() const { return occupancy_weight_; }
    double get_bandwidth_weight() const { return bandwidth_weight_; }
    
    // Setters
    void set_optimization_strategy(const std::string& strategy);
    void set_optimization_weights(double occupancy_weight, double bandwidth_weight);
    
protected:
    // Métodos de utilidad comunes
    int find_optimal_block_size_range(int min_size, int max_size, 
                                    const DataStructureInfo& data_info) const;
    double calculate_occupancy_score(int block_size, const DataStructureInfo& data_info) const;
    double calculate_bandwidth_score(const DataStructureInfo& data_info) const;
    
    // Logging helpers
    void log_optimization_start(const DataStructureInfo& data_info) const;
    void log_optimization_complete(const LaunchConfig& config, double time_us) const;
    void log_performance_metrics(const std::string& metric_name, double value) const;
};

} // namespace cato

#endif // BASE_LAUNCH_OPTIMIZER_CUH
