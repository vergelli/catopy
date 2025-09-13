// include/device/KernelOptimizerConstants.cuh
#ifndef KERNEL_OPTIMIZER_CONSTANTS_CUH
#define KERNEL_OPTIMIZER_CONSTANTS_CUH

namespace KernelOptimizerConstants {
    // Optimization weights
    constexpr double DEFAULT_OCCUPANCY_WEIGHT = 0.4;
    constexpr double DEFAULT_BANDWIDTH_WEIGHT = 0.4;
    constexpr double DEFAULT_LATENCY_WEIGHT = 0.2;

    // Strategy names
    constexpr const char* STRATEGY_OCCUPANCY = "occupancy";
    constexpr const char* STRATEGY_BANDWIDTH = "bandwidth";
    constexpr const char* STRATEGY_BALANCED = "balanced";
    constexpr const char* STRATEGY_ND_FLATTENING = "nd_flattening";

    // Dimensionality
    constexpr int DIM_1D = 1;
    constexpr int DIM_2D = 2;
    constexpr int DIM_3D = 3;
    constexpr int DIM_ND = -1; // For N-dimensional tensors

    // Heuristics
    constexpr int MIN_BLOCK_SIZE = 32;  // Minimum viable block size
    constexpr int MAX_BLOCK_SIZE = 1024; // CUDA limit
    constexpr double OCCUPANCY_THRESHOLD = 0.75; // Target occupancy

    // High-dimensionality constants
    constexpr int MAX_SUPPORTED_DIMENSIONS = 10; // Reasonable limit
    constexpr double HIGH_DIM_PENALTY_FACTOR = 0.1; // Penalty per extra dimension
    constexpr int HIGH_DIM_BLOCK_SIZE_LIMIT = 256; // Smaller blocks for high-dim

    // Logging messages
    constexpr const char* LOG_OPTIMIZER_INIT = "KernelLaunchOptimizer initialized for {}D operations";
    constexpr const char* LOG_OPTIMIZATION_START = "Starting {}D optimization for {} elements";
    constexpr const char* LOG_OPTIMIZATION_COMPLETE = "{}D optimization completed in {} μs";
    constexpr const char* LOG_TO_BE_IMPLEMENTED = "TO BE IMPLEMENTED: {}D optimization not yet available";
    constexpr const char* LOG_HIGH_DIMENSIONALITY = "High-dimensionality tensor detected ({}D), using flattening strategy";
}

#endif // KERNEL_OPTIMIZER_CONSTANTS_CUH