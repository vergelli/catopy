// src/device/optimizers/VectorLaunchOptimizer.cu
#include "../../../include/device/optimizers/VectorLaunchOptimizer.cuh"
#include <algorithm>
#include <cmath>
#include <chrono>

namespace cato {

/**
 * @brief Constructs a VectorLaunchOptimizer for 1D vector operations
 * 
 * Initializes the vector-specific optimizer with balanced optimization strategy
 * and default weight configuration. This optimizer is specifically designed
 * for 1D vector operations and uses vector-specific optimization algorithms.
 * 
 * @param device CUDA device information for optimization calculations
 * 
 * @note Uses balanced optimization strategy by default
 * @note Configured with default occupancy and bandwidth weights
 * @note Specialized for 1D vector data structures
 * @note Inherits base optimization functionality from BaseLaunchOptimizer
 */
VectorLaunchOptimizer::VectorLaunchOptimizer(const Device& device)
    : BaseLaunchOptimizer(device, KernelOptimizerConstants::STRATEGY_BALANCED,
                         KernelOptimizerConstants::DEFAULT_OCCUPANCY_WEIGHT,
                         KernelOptimizerConstants::DEFAULT_BANDWIDTH_WEIGHT) {
    
    Logger::debug("VectorLaunchOptimizer initialized for 1D operations");
}

/**
 * @brief Optimize launch configuration for vector data structure
 * 
 * Performs optimization for 1D vector data structures by validating input
 * and delegating to the vector-specific optimization algorithm. This is the
 * main entry point for vector optimization from the base class interface.
 * 
 * @param data_info Data structure information to optimize for
 * @return Optimized launch configuration for the vector
 * @throws std::invalid_argument if data is not 1D or not a vector type
 * 
 * @note Validates that input data is 1D and of vector type
 * @note Delegates to optimize_for_operation for actual optimization
 * @note Throws exceptions for invalid data structure types or dimensions
 * @note Returns optimized configuration with vector-specific parameters
 */
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

/**
 * @brief Optimize launch configuration for specific vector operation
 * 
 * Performs comprehensive optimization for vector operations by calculating
 * optimal block size, creating 1D launch configuration, and applying
 * vector-specific optimizations. This is the core optimization algorithm
 * for vector operations.
 * 
 * @param vector_info Vector data structure information
 * @param operation_type Type of operation being optimized (e.g., "add", "mul")
 * @return Optimized launch configuration with vector-specific parameters
 * 
 * @note Calculates optimal block size using vector-specific algorithms
 * @note Creates 1D grid and block configuration
 * @note Applies vector-specific optimization tags for tracking
 * @note Logs optimization process for debugging and monitoring
 * @note Returns configuration with occupancy and bandwidth metrics
 */
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

/**
 * @brief Analyze occupancy for vector operations
 * 
 * Delegates to vector-specific occupancy analysis with generic parameters.
 * This method provides the base class interface for occupancy analysis
 * while using vector-optimized algorithms internally.
 * 
 * @param block_size Thread block size to analyze
 * @return Occupancy ratio (0.0 to 1.0, higher is better)
 * 
 * @note Uses vector-specific occupancy analysis with generic size parameter
 * @note Delegates to analyze_vector_occupancy for actual calculation
 * @note Provides base class interface compatibility
 * @note Used by base class optimization algorithms
 */
double VectorLaunchOptimizer::analyze_occupancy(int block_size) const {
    return analyze_vector_occupancy(block_size, 0); // Tamaño 0 para análisis genérico
}

/**
 * @brief Estimate bandwidth utilization for vector data structure
 * 
 * Validates that the data structure is a vector type and delegates to
 * vector-specific bandwidth estimation. This method provides the base class
 * interface while using vector-optimized bandwidth calculations.
 * 
 * @param data_info Data structure information to analyze
 * @return Bandwidth utilization ratio (0.0 to 1.0, higher is better)
 * @throws std::invalid_argument if data structure is not a vector type
 * 
 * @note Validates that input is a VectorInfo type
 * @note Delegates to estimate_vector_bandwidth for actual calculation
 * @note Provides base class interface compatibility
 * @note Used by base class optimization algorithms
 */
double VectorLaunchOptimizer::estimate_bandwidth_utilization(const DataStructureInfo& data_info) const {
    const VectorInfo* vector_info = dynamic_cast<const VectorInfo*>(&data_info);
    if (!vector_info) {
        throw std::invalid_argument("VectorLaunchOptimizer: Invalid data structure type");
    }
    
    return estimate_vector_bandwidth(*vector_info);
}

/**
 * @brief Analyze occupancy for vector operations with specific size
 * 
 * Calculates the theoretical occupancy for vector operations based on the
 * thread block size and device capabilities. This provides vector-specific
 * occupancy analysis that considers the 1D nature of vector operations.
 * 
 * @param block_size Thread block size to analyze
 * @param vector_size Size of the vector (currently unused, reserved for future use)
 * @return Occupancy ratio (0.0 to 1.0, higher is better)
 * 
 * @note Calculates active threads per SM based on block size
 * @note Uses device maximum threads per multiprocessor for normalization
 * @note Provides vector-specific occupancy analysis
 * @note Higher occupancy generally indicates better GPU utilization
 */
double VectorLaunchOptimizer::analyze_vector_occupancy(int block_size, size_t vector_size) const {
    int max_threads_per_sm = device_.get_max_threads_per_multiprocessor();
    int active_blocks_per_sm = max_threads_per_sm / block_size;
    int active_threads_per_sm = active_blocks_per_sm * block_size;
    
    return static_cast<double>(active_threads_per_sm) / max_threads_per_sm;
}

/**
 * @brief Estimate bandwidth utilization for vector operations
 * 
 * Calculates the estimated memory bandwidth utilization for vector operations
 * based on the vector size, element size, and device characteristics. This
 * provides vector-specific bandwidth analysis for optimization decisions.
 * 
 * @param vector_info Vector data structure information
 * @return Bandwidth utilization ratio (0.0 to 1.0, higher is better)
 * 
 * @note Uses simplified heuristic for device bandwidth estimation (500 GB/s)
 * @note Considers element-wise operations (2 reads + 1 write per element)
 * @note Includes 20% overhead factor for realistic kernel execution time
 * @note Higher utilization indicates better memory bandwidth efficiency
 * @note Used in optimization scoring to balance occupancy and bandwidth
 */
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

/**
 * @brief Find optimal block size for 1D vector operations
 * 
 * Searches for the optimal thread block size for vector operations by evaluating
 * different block sizes that are multiples of the warp size. Uses vector-specific
 * scoring that considers both occupancy and bandwidth utilization.
 * 
 * @param vector_info Vector data structure information for optimization
 * @return Optimal thread block size for vector operations
 * 
 * @note Only considers block sizes that are multiples of the warp size
 * @note Searches from warp_size to max_threads_per_block
 * @note Uses vector-specific scoring algorithm for evaluation
 * @note Returns the block size with the highest combined score
 * @note Falls back to warp_size if no better option is found
 */
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

/**
 * @brief Create 1D launch configuration for vector operations
 * 
 * Creates a complete launch configuration optimized for 1D vector operations
 * with the specified block size. Calculates grid dimensions, occupancy metrics,
 * and performance estimates specific to vector operations.
 * 
 * @param vector_info Vector data structure information
 * @param block_size Optimal thread block size for the configuration
 * @return Complete launch configuration for vector operations
 * 
 * @note Creates 1D grid and block dimensions (y=1, z=1)
 * @note Calculates grid size based on total elements and block size
 * @note Computes vector-specific occupancy and bandwidth metrics
 * @note Estimates overall performance using weighted combination
 * @note Sets dimensionality to 1D for vector operations
 */
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

/**
 * @brief Calculate optimization score for vector block size
 * 
 * Computes a vector-specific optimization score that combines occupancy,
 * bandwidth utilization, and a logarithmic factor that favors larger
 * block sizes. This scoring function is tailored for vector operations.
 * 
 * @param block_size Thread block size to evaluate
 * @param vector_info Vector data structure information
 * @return Combined optimization score (higher is better)
 * 
 * @note Combines occupancy and bandwidth using configured weights
 * @note Includes logarithmic factor to favor larger block sizes
 * @note Logarithmic factor provides diminishing returns for very large blocks
 * @note Used by find_optimal_block_size_1d for block size comparison
 * @note Higher scores indicate better expected performance
 */
double VectorLaunchOptimizer::calculate_vector_score(int block_size, const VectorInfo& vector_info) const {
    double occupancy = analyze_vector_occupancy(block_size, vector_info.get_total_elements());
    double bandwidth = estimate_vector_bandwidth(vector_info);
    
    // Score específico para vectores: occupancy + log factor para favorecer bloques más grandes
    double log_factor = std::log2(block_size / device_.get_warp_size() + 1);
    
    return occupancy * occupancy_weight_ + bandwidth * bandwidth_weight_ + log_factor * 0.1;
}

/**
 * @brief Log the start of vector optimization process
 * 
 * Outputs debug information about the beginning of vector-specific optimization,
 * including the number of elements to be processed for debugging and monitoring.
 * 
 * @param vector_info Vector data structure information to log
 * 
 * @note Only outputs when debug logging is enabled
 * @note Provides context for vector optimization process monitoring
 * @note Used for debugging vector optimization performance and behavior
 * @note Logs element count for optimization context
 */
void VectorLaunchOptimizer::log_vector_optimization_start(const VectorInfo& vector_info) const {
    Logger::debug("Vector optimization started for {} elements", 
                 vector_info.get_total_elements());
}

/**
 * @brief Log the completion of vector optimization process
 * 
 * Outputs debug information about the completion of vector-specific optimization,
 * including occupancy metrics, bandwidth utilization, and the resulting
 * grid and block dimensions for debugging and monitoring.
 * 
 * @param config Resulting launch configuration to log
 * 
 * @note Only outputs when debug logging is enabled
 * @note Provides detailed optimization results for analysis
 * @note Logs occupancy and bandwidth utilization metrics
 * @note Shows final grid and block dimensions for verification
 * @note Used for debugging vector optimization effectiveness
 */
void VectorLaunchOptimizer::log_vector_optimization_complete(const LaunchConfig& config) const {
    Logger::debug("Vector Occupancy: {:.4f}", config.occupancy);
    Logger::debug("Vector Bandwidth Util: {:.4f}", config.memory_bandwidth_utilization);
    Logger::debug("Vector optimization completed - Grid: ({}, {}, {}), Block: ({}, {}, {})", 
                 config.gridDim.x, config.gridDim.y, config.gridDim.z,
                 config.blockDim.x, config.blockDim.y, config.blockDim.z);
}

} // namespace cato
