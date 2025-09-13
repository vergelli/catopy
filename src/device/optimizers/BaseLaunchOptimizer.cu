// src/device/optimizers/BaseLaunchOptimizer.cu
#include "../../../include/device/optimizers/BaseLaunchOptimizer.cuh" // Para definiciones completas
#include <algorithm>
#include <cmath>
#include <chrono>

namespace cato {

/**
 * @brief Constructs a BaseLaunchOptimizer with specified optimization parameters
 * 
 * Initializes the base optimizer with device information and optimization strategy.
 * This constructor sets up the fundamental optimization parameters that will be used
 * by derived classes for kernel launch optimization.
 * 
 * @param device CUDA device information for optimization calculations
 * @param strategy Optimization strategy identifier (e.g., "balanced", "occupancy", "bandwidth")
 * @param occupancy_weight Weight factor for occupancy-based optimization (0.0 to 1.0)
 * @param bandwidth_weight Weight factor for bandwidth-based optimization (0.0 to 1.0)
 * 
 * @note The sum of occupancy_weight and bandwidth_weight should typically equal 1.0
 * @note Strategy determines the overall optimization approach used by derived classes
 * @note Weights control the balance between occupancy and bandwidth optimization
 */
BaseLaunchOptimizer::BaseLaunchOptimizer(const Device& device, 
                                       const std::string& strategy,
                                       double occupancy_weight,
                                       double bandwidth_weight)
    : device_(device)
    , optimization_strategy_(strategy)
    , occupancy_weight_(occupancy_weight)
    , bandwidth_weight_(bandwidth_weight) {
    
    Logger::debug("BaseLaunchOptimizer initialized with strategy: {}", strategy);
}

/**
 * @brief Get the optimal warp size for the current device
 * 
 * Returns the native warp size of the CUDA device, which is fundamental
 * for optimal thread block sizing in CUDA kernels.
 * 
 * @return Optimal warp size for the device (typically 32 for modern GPUs)
 * 
 * @note Warp size is device-specific and affects occupancy calculations
 * @note Thread block sizes should be multiples of the warp size for optimal performance
 */
int BaseLaunchOptimizer::get_optimal_warp_size() const {
    return device_.get_warp_size();
}

/**
 * @brief Calculate the number of thread blocks needed for given data size
 * 
 * Computes the minimum number of thread blocks required to process all elements
 * using the specified block size. Uses ceiling division to ensure all elements
 * are covered by the grid.
 * 
 * @param block_size Number of threads per block
 * @param total_elements Total number of elements to process
 * @return Number of thread blocks needed
 * 
 * @note Uses ceiling division: ceil(total_elements / block_size)
 * @note Ensures complete coverage of all data elements
 * @note Thread blocks may process more elements than needed (handled by bounds checking in kernels)
 */
int BaseLaunchOptimizer::calculate_blocks_needed(int block_size, size_t total_elements) const {
    return (total_elements + block_size - 1) / block_size;
}

/**
 * @brief Validate a launch configuration against device limits
 * 
 * Performs comprehensive validation of a LaunchConfig to ensure it meets
 * device constraints and CUDA launch requirements. Checks thread block size
 * limits and configuration validity.
 * 
 * @param config Launch configuration to validate
 * @return true if configuration is valid, false otherwise
 * 
 * @note Validates that total threads per block does not exceed device maximum
 * @note Checks configuration validity flag
 * @note Logs error messages for invalid configurations
 * @note Returns false immediately if config.is_valid is false
 */
bool BaseLaunchOptimizer::validate_launch_config(const LaunchConfig& config) const {
    if (!config.is_valid) {
        return false;
    }
    
    // Verificar límites del dispositivo
    int max_threads_per_block = device_.get_max_threads_per_block();
    int total_threads = config.blockDim.x * config.blockDim.y * config.blockDim.z;
    
    if (total_threads > max_threads_per_block) {
        Logger::error("Block size {} exceeds device limit {}", 
                     total_threads, max_threads_per_block);
        return false;
    }
    
    return true;
}

/**
 * @brief Set the optimization strategy for kernel launch optimization
 * 
 * Updates the optimization strategy used by the optimizer. This affects
 * how the optimizer balances different performance factors when selecting
 * optimal launch parameters.
 * 
 * @param strategy New optimization strategy identifier
 * 
 * @note Common strategies include "balanced", "occupancy", "bandwidth"
 * @note Strategy change takes effect for subsequent optimization calls
 * @note Logs the strategy change for debugging purposes
 */
void BaseLaunchOptimizer::set_optimization_strategy(const std::string& strategy) {
    optimization_strategy_ = strategy;
    Logger::debug("Optimization strategy changed to: {}", strategy);
}

/**
 * @brief Set optimization weights for performance factor balancing
 * 
 * Updates the weight factors used in the optimization scoring function.
 * These weights control how much influence occupancy and bandwidth utilization
 * have on the final optimization decision.
 * 
 * @param occupancy_weight Weight for occupancy-based optimization (0.0 to 1.0)
 * @param bandwidth_weight Weight for bandwidth-based optimization (0.0 to 1.0)
 * 
 * @note Weights should typically sum to 1.0 for balanced optimization
 * @note Higher occupancy_weight favors thread block sizes that maximize occupancy
 * @note Higher bandwidth_weight favors configurations that maximize memory bandwidth
 * @note Weight changes take effect for subsequent optimization calls
 */
void BaseLaunchOptimizer::set_optimization_weights(double occupancy_weight, double bandwidth_weight) {
    occupancy_weight_ = occupancy_weight;
    bandwidth_weight_ = bandwidth_weight;
    Logger::debug("Optimization weights updated - occupancy: {}, bandwidth: {}", 
                 occupancy_weight, bandwidth_weight);
}

/**
 * @brief Find optimal block size within specified range
 * 
 * Searches for the optimal thread block size within the given range by evaluating
 * different block sizes and selecting the one with the highest optimization score.
 * Only considers block sizes that are multiples of the warp size for optimal performance.
 * 
 * @param min_size Minimum block size to consider
 * @param max_size Maximum block size to consider
 * @param data_info Data structure information for optimization calculations
 * @return Optimal block size within the specified range
 * 
 * @note Only evaluates block sizes that are multiples of the device warp size
 * @note Uses occupancy and bandwidth scoring to determine optimal size
 * @note Returns the block size with the highest combined score
 * @note Falls back to warp_size if no valid block size is found
 */
int BaseLaunchOptimizer::find_optimal_block_size_range(int min_size, int max_size, 
                                                     const DataStructureInfo& data_info) const {
    int warp_size = get_optimal_warp_size();
    int best_block_size = warp_size;
    double best_score = 0.0;

    // Buscar en múltiplos del warp size
    for (int block_size = min_size; block_size <= max_size; block_size += warp_size) {
        double score = calculate_occupancy_score(block_size, data_info);
        
        if (score > best_score) {
            best_score = score;
            best_block_size = block_size;
        }
    }
    
    return best_block_size;
}

/**
 * @brief Calculate combined optimization score for a block size
 * 
 * Computes a weighted combination of occupancy and bandwidth scores to evaluate
 * the overall performance potential of a given thread block size. This score
 * is used to compare different block size options during optimization.
 * 
 * @param block_size Thread block size to evaluate
 * @param data_info Data structure information for bandwidth calculations
 * @return Combined optimization score (higher is better)
 * 
 * @note Score combines occupancy and bandwidth factors using configured weights
 * @note Higher scores indicate better expected performance
 * @note Weights are applied according to current optimization strategy
 * @note Used internally by find_optimal_block_size_range for comparison
 */
double BaseLaunchOptimizer::calculate_occupancy_score(int block_size, const DataStructureInfo& data_info) const {
    double occupancy = analyze_occupancy(block_size);
    double bandwidth = calculate_bandwidth_score(data_info);
    
    return occupancy * occupancy_weight_ + bandwidth * bandwidth_weight_;
}

/**
 * @brief Calculate bandwidth utilization score for data structure
 * 
 * Computes a bandwidth utilization score based on the data structure characteristics.
 * This score estimates how efficiently the memory bandwidth will be utilized
 * for the given data structure and operation type.
 * 
 * @param data_info Data structure information for bandwidth analysis
 * @return Bandwidth utilization score (0.0 to 1.0, higher is better)
 * 
 * @note Delegates to estimate_bandwidth_utilization for actual calculation
 * @note Score is used in combined optimization scoring
 * @note Higher scores indicate better memory bandwidth utilization
 * @note Used internally by calculate_occupancy_score
 */
double BaseLaunchOptimizer::calculate_bandwidth_score(const DataStructureInfo& data_info) const {
    return estimate_bandwidth_utilization(data_info);
}

/**
 * @brief Log the start of optimization process
 * 
 * Outputs debug information about the beginning of the optimization process,
 * including data dimensionality and element count for debugging and monitoring.
 * 
 * @param data_info Data structure information to log
 * 
 * @note Only outputs when debug logging is enabled
 * @note Provides context for optimization process monitoring
 * @note Used for debugging optimization performance and behavior
 */
void BaseLaunchOptimizer::log_optimization_start(const DataStructureInfo& data_info) const {
    Logger::debug("Starting {}D optimization for {} elements", 
                 data_info.get_dimensionality(), 
                 data_info.get_total_elements());
}

/**
 * @brief Log the completion of optimization process
 * 
 * Outputs debug information about the completion of the optimization process,
 * including the resulting configuration dimensionality and optimization time.
 * 
 * @param config Resulting launch configuration
 * @param time_us Optimization time in microseconds
 * 
 * @note Only outputs when debug logging is enabled
 * @note Provides performance metrics for optimization monitoring
 * @note Used for debugging optimization efficiency and timing
 */
void BaseLaunchOptimizer::log_optimization_complete(const LaunchConfig& config, double time_us) const {
    Logger::debug("{}D optimization completed in {:.3f} μs", 
                 config.dimensionality, time_us);
}

/**
 * @brief Log performance metrics for debugging and analysis
 * 
 * Outputs debug information about specific performance metrics during
 * optimization, providing detailed insights into optimization decisions.
 * 
 * @param metric_name Name of the performance metric
 * @param value Value of the performance metric
 * 
 * @note Only outputs when debug logging is enabled
 * @note Used for detailed optimization analysis and debugging
 * @note Provides formatted output with 4 decimal precision
 * @note Common metrics include occupancy, bandwidth, and combined scores
 */
void BaseLaunchOptimizer::log_performance_metrics(const std::string& metric_name, double value) const {
    Logger::debug("Performance metric - {}: {:.4f}", metric_name, value);
}

} // namespace cato
