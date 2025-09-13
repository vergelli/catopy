// include/device/optimizers/VectorLaunchOptimizer.cuh
#ifndef VECTOR_LAUNCH_OPTIMIZER_CUH
#define VECTOR_LAUNCH_OPTIMIZER_CUH

#include "BaseLaunchOptimizer.cuh"
#include "../KernelOptimizerConstants.cuh"
#include <memory>

namespace cato {

// Forward declarations
class VectorInfo;

/**
 * @brief Optimizador especializado para operaciones de vectores 1D
 * 
 * Este optimizador está específicamente diseñado para operaciones element-wise
 * en vectores 1D, aprovechando patrones de acceso secuencial y optimizaciones
 * específicas para este tipo de datos.
 */
class VectorLaunchOptimizer : public BaseLaunchOptimizer {
public:
    explicit VectorLaunchOptimizer(const Device& device);
    ~VectorLaunchOptimizer() = default;

    // Implementación de métodos virtuales puros
    LaunchConfig optimize(const DataStructureInfo& data_info) override;
    double analyze_occupancy(int block_size) const override;
    double estimate_bandwidth_utilization(const DataStructureInfo& data_info) const override;

    // Métodos específicos para vectores
    LaunchConfig optimize_for_operation(const VectorInfo& vector_info, 
                                      const std::string& operation_type = "elementwise");

    // Análisis específico para vectores
    double analyze_vector_occupancy(int block_size, size_t vector_size) const;
    double estimate_vector_bandwidth(const VectorInfo& vector_info) const;

    // Optimización específica para operaciones element-wise
    int find_optimal_block_size_1d(const VectorInfo& vector_info) const;

private:
    // Constantes específicas para vectores
    static constexpr int VECTOR_OPTIMAL_BLOCK_SIZE = 256;
    static constexpr double VECTOR_OCCUPANCY_TARGET = 0.75;
    static constexpr double VECTOR_BANDWIDTH_FACTOR = 1.0;

    // Métodos de optimización específicos
    LaunchConfig create_1d_config(const VectorInfo& vector_info, int block_size) const;
    double calculate_vector_score(int block_size, const VectorInfo& vector_info) const;

    // Logging específico para vectores
    void log_vector_optimization_start(const VectorInfo& vector_info) const;
    void log_vector_optimization_complete(const LaunchConfig& config) const;
};

} // namespace cato

#endif // VECTOR_LAUNCH_OPTIMIZER_CUH
