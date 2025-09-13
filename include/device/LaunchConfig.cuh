// include/device/LaunchConfig.cuh
#ifndef LAUNCH_CONFIG_CUH
#define LAUNCH_CONFIG_CUH

#include <cuda_runtime.h>

namespace cato {

/**
 * @brief Configuration for CUDA kernel launch parameters
 * 
 * This structure contains the optimized grid and block dimensions
 * for launching CUDA kernels, along with additional parameters
 * like stream and shared memory configuration.
 */
struct LaunchConfig {
    dim3 gridDim;           ///< Grid dimensions (number of blocks)
    dim3 blockDim;          ///< Block dimensions (threads per block)
    cudaStream_t stream;    ///< CUDA stream for asynchronous execution
    size_t shared_mem;      ///< Shared memory per block in bytes
    
    // Additional fields for optimization tracking
    int dimensionality;     ///< Data structure dimensionality
    std::string optimization_strategy; ///< Strategy used for optimization
    double occupancy;       ///< Achieved occupancy (0.0 to 1.0)
    double memory_bandwidth_utilization; ///< Memory bandwidth utilization (0.0 to 1.0)
    double estimated_performance; ///< Estimated performance score
    std::vector<std::string> applied_optimizations; ///< List of applied optimizations
    bool is_valid;          ///< Whether the configuration is valid
    
    /**
     * @brief Default constructor
     */
    LaunchConfig() : gridDim(1, 1, 1), blockDim(1, 1, 1), stream(0), shared_mem(0),
                     dimensionality(1), optimization_strategy("balanced"), 
                     occupancy(0.0), memory_bandwidth_utilization(0.0), 
                     estimated_performance(0.0), is_valid(false) {}
    
    /**
     * @brief Constructor with parameters
     * @param grid Grid dimensions
     * @param block Block dimensions
     * @param str CUDA stream (default: 0)
     * @param smem Shared memory per block (default: 0)
     */
    LaunchConfig(dim3 grid, dim3 block, cudaStream_t str = 0, size_t smem = 0)
        : gridDim(grid), blockDim(block), stream(str), shared_mem(smem),
          dimensionality(1), optimization_strategy("balanced"), 
          occupancy(0.0), memory_bandwidth_utilization(0.0), 
          estimated_performance(0.0), is_valid(true) {}
};

} // namespace cato

#endif // LAUNCH_CONFIG_CUH
