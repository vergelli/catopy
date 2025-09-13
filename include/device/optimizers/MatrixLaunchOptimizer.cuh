// include/device/optimizers/MatrixLaunchOptimizer.cuh
#ifndef MATRIX_LAUNCH_OPTIMIZER_CUH
#define MATRIX_LAUNCH_OPTIMIZER_CUH

#include "BaseLaunchOptimizer.cuh"
#include "../KernelOptimizerConstants.cuh"

namespace cato {

// Forward declarations
class MatrixInfo;

/**
 * @brief Optimizador especializado para operaciones de matrices 2D
 * 
 * PLACEHOLDER - Esta clase será implementada en el futuro para operaciones
 * de matrices 2D como multiplicación de matrices, transposición, etc.
 */
class MatrixLaunchOptimizer : public BaseLaunchOptimizer {
public:
    explicit MatrixLaunchOptimizer(const Device& device);
    ~MatrixLaunchOptimizer() = default;
    
    // Implementación de métodos virtuales puros
    LaunchConfig optimize(const DataStructureInfo& data_info) override;
    double analyze_occupancy(int block_size) const override;
    double estimate_bandwidth_utilization(const DataStructureInfo& data_info) const override;
    
    // Métodos específicos para matrices (PLACEHOLDER)
    LaunchConfig optimize_for_operation(const MatrixInfo& matrix_info, 
                                      const std::string& operation_type = "matmul");
    
private:
    // Constantes específicas para matrices (PLACEHOLDER)
    static constexpr int MATRIX_OPTIMAL_BLOCK_SIZE = 16; // 16x16 blocks
    static constexpr double MATRIX_OCCUPANCY_TARGET = 0.75;
    static constexpr double MATRIX_CACHE_FACTOR = 1.2; // Consideración de cache locality
};

} // namespace cato

#endif // MATRIX_LAUNCH_OPTIMIZER_CUH
