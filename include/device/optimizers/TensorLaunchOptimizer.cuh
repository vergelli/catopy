// include/device/optimizers/TensorLaunchOptimizer.cuh
#ifndef TENSOR_LAUNCH_OPTIMIZER_CUH
#define TENSOR_LAUNCH_OPTIMIZER_CUH

#include "BaseLaunchOptimizer.cuh"
#include "../KernelOptimizerConstants.cuh"

namespace cato {

// Forward declarations
class Tensor3DInfo;
class TensorNDInfo;

/**
 * @brief Optimizador especializado para operaciones de tensores 3D+
 * 
 * PLACEHOLDER - Esta clase será implementada en el futuro para operaciones
 * de tensores multidimensionales como contracciones, reducciones, etc.
 */
class TensorLaunchOptimizer : public BaseLaunchOptimizer {
public:
    explicit TensorLaunchOptimizer(const Device& device);
    ~TensorLaunchOptimizer() = default;
    
    // Implementación de métodos virtuales puros
    LaunchConfig optimize(const DataStructureInfo& data_info) override;
    double analyze_occupancy(int block_size) const override;
    double estimate_bandwidth_utilization(const DataStructureInfo& data_info) const override;
    
    // Métodos específicos para tensores (PLACEHOLDER)
    LaunchConfig optimize_for_3d_operation(const Tensor3DInfo& tensor_info, 
                                         const std::string& operation_type = "contraction");
    LaunchConfig optimize_for_nd_operation(const TensorNDInfo& tensor_info, 
                                         const std::string& operation_type = "reduction");
    
private:
    // Constantes específicas para tensores (PLACEHOLDER)
    static constexpr int TENSOR_OPTIMAL_BLOCK_SIZE = 128; // Bloques más pequeños para alta dimensionalidad
    static constexpr double TENSOR_OCCUPANCY_TARGET = 0.70; // Objetivo más conservador
    static constexpr double TENSOR_DIMENSIONALITY_PENALTY = 0.1; // Penalización por dimensionalidad
};

} // namespace cato

#endif // TENSOR_LAUNCH_OPTIMIZER_CUH
