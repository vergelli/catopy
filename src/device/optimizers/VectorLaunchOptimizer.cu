// src/device/optimizers/VectorLaunchOptimizer.cu
#include "../../../include/device/optimizers/VectorLaunchOptimizer.cuh"
#include <algorithm>
#include <cmath>
#include <chrono>

namespace cato {

VectorLaunchOptimizer::VectorLaunchOptimizer(const Device& device)
    : BaseLaunchOptimizer(device, KernelOptimizerConstants::STRATEGY_BALANCED,
                         KernelOptimizerConstants::DEFAULT_OCCUPANCY_WEIGHT,
                         KernelOptimizerConstants::DEFAULT_BANDWIDTH_WEIGHT) {
    
    Logger::debug("VectorLaunchOptimizer initialized for 1D operations");
}

LaunchConfig VectorLaunchOptimizer::optimize(const DataStructureInfo& data_info) {
    // Verificar que es un vector
    if (data_info.get_dimensionality() != KernelOptimizerConstants::DIM_1D) {
        throw std::invalid_argument("VectorLaunchOptimizer: Expected 1D data, got " + 
                                  std::to_string(data_info.get_dimensionality()) + "D");
    }
    
    // Convertir a VectorInfo específico
    const VectorInfo* vector_info = dynamic_cast<const VectorInfo*>(&data_info);
    if (!vector_info) {
        throw std::invalid_argument("VectorLaunchOptimizer: Invalid data structure type");
    }
    
    return optimize_for_operation(*vector_info);
}

LaunchConfig VectorLaunchOptimizer::optimize_for_operation(const VectorInfo& vector_info, 
                                                         const std::string& operation_type) {
    log_vector_optimization_start(vector_info);
    
    // Calcular tamaño óptimo de bloque específico para vectores
    int optimal_block_size = find_optimal_block_size_1d(vector_info);
    
    // Crear configuración optimizada
    LaunchConfig config = create_1d_config(vector_info, optimal_block_size);
    
    // Aplicar optimizaciones específicas para vectores
    config.applied_optimizations.push_back("vector_1d_optimization");
    config.applied_optimizations.push_back("optimal_block_size_" + std::to_string(optimal_block_size));
    config.applied_optimizations.push_back("operation_" + operation_type);
    
    log_vector_optimization_complete(config);
    
    return config;
}

double VectorLaunchOptimizer::analyze_occupancy(int block_size) const {
    return analyze_vector_occupancy(block_size, 0); // Tamaño 0 para análisis genérico
}

double VectorLaunchOptimizer::estimate_bandwidth_utilization(const DataStructureInfo& data_info) const {
    const VectorInfo* vector_info = dynamic_cast<const VectorInfo*>(&data_info);
    if (!vector_info) {
        throw std::invalid_argument("VectorLaunchOptimizer: Invalid data structure type");
    }
    
    return estimate_vector_bandwidth(*vector_info);
}

double VectorLaunchOptimizer::analyze_vector_occupancy(int block_size, size_t vector_size) const {
    int max_threads_per_sm = device_.get_max_threads_per_multiprocessor();
    int active_blocks_per_sm = max_threads_per_sm / block_size;
    int active_threads_per_sm = active_blocks_per_sm * block_size;
    
    return static_cast<double>(active_threads_per_sm) / max_threads_per_sm;
}

double VectorLaunchOptimizer::estimate_vector_bandwidth(const VectorInfo& vector_info) const {
    size_t total_elements = vector_info.get_total_elements();
    size_t element_size = vector_info.get_element_size_bytes();
    
    // Estimación de ancho de banda del dispositivo
    double estimated_bandwidth_gb_s = 500.0; // Heurística simplificada
    double estimated_bandwidth_bytes_s = estimated_bandwidth_gb_s * 1e9;
    
    // Calcular utilización para operaciones element-wise (2 reads + 1 write)
    size_t total_memory_bytes = total_elements * element_size * 3;
    double theoretical_time_s = total_memory_bytes / estimated_bandwidth_bytes_s;
    double estimated_kernel_time_s = theoretical_time_s * 1.2; // 20% overhead
    
    return theoretical_time_s / estimated_kernel_time_s;
}

int VectorLaunchOptimizer::find_optimal_block_size_1d(const VectorInfo& vector_info) const {
    int warp_size = device_.get_warp_size();
    int max_threads_per_block = device_.get_max_threads_per_block();
    
    int best_block_size = warp_size;
    double best_score = 0.0;
    
    // Buscar en múltiplos del warp size
    for (int block_size = warp_size; block_size <= max_threads_per_block; block_size += warp_size) {
        double score = calculate_vector_score(block_size, vector_info);
        
        if (score > best_score) {
            best_score = score;
            best_block_size = block_size;
        }
    }
    
    return best_block_size;
}

LaunchConfig VectorLaunchOptimizer::create_1d_config(const VectorInfo& vector_info, int block_size) const {
    LaunchConfig config;
    config.dimensionality = KernelOptimizerConstants::DIM_1D;
    config.optimization_strategy = optimization_strategy_;
    
    // Calcular dimensiones de grilla
    size_t total_elements = vector_info.get_total_elements();
    int blocks_needed = calculate_blocks_needed(block_size, total_elements);
    
    // Configurar dimensiones 1D
    config.blockDim = dim3(block_size, 1, 1);
    config.gridDim = dim3(blocks_needed, 1, 1);
    
    // Calcular métricas específicas para vectores
    config.occupancy = analyze_vector_occupancy(block_size, total_elements);
    config.memory_bandwidth_utilization = estimate_vector_bandwidth(vector_info);
    config.estimated_performance = config.occupancy * occupancy_weight_ + 
                                 config.memory_bandwidth_utilization * bandwidth_weight_;
    
    return config;
}

double VectorLaunchOptimizer::calculate_vector_score(int block_size, const VectorInfo& vector_info) const {
    double occupancy = analyze_vector_occupancy(block_size, vector_info.get_total_elements());
    double bandwidth = estimate_vector_bandwidth(vector_info);
    
    // Score específico para vectores: occupancy + log factor para favorecer bloques más grandes
    double log_factor = std::log2(block_size / device_.get_warp_size() + 1);
    
    return occupancy * occupancy_weight_ + bandwidth * bandwidth_weight_ + log_factor * 0.1;
}

void VectorLaunchOptimizer::log_vector_optimization_start(const VectorInfo& vector_info) const {
    Logger::debug("Vector optimization started for {} elements", 
                 vector_info.get_total_elements());
}

void VectorLaunchOptimizer::log_vector_optimization_complete(const LaunchConfig& config) const {
    Logger::debug("Vector Occupancy: {:.4f}", config.occupancy);
    Logger::debug("Vector Bandwidth Util: {:.4f}", config.memory_bandwidth_utilization);
    Logger::debug("Vector optimization completed - Grid: ({}, {}, {}), Block: ({}, {}, {})", 
                 config.gridDim.x, config.gridDim.y, config.gridDim.z,
                 config.blockDim.x, config.blockDim.y, config.blockDim.z);
}

} // namespace cato
