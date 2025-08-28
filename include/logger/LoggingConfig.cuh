#ifndef LOGGING_CONFIG_CUH
#define LOGGING_CONFIG_CUH

#include "Logger.cuh"
#include <string>

/**
 * @file LoggingConfig.cuh
 * @brief Configuration utilities for the logging system
 * 
 * This file provides:
 * - Environment-based logging configuration
 * - Easy setup for different build types
 * - CUDA-specific logging presets
 */

namespace LoggingConfig {

    /**
     * @brief Initialize logger based on environment variables
     * 
     * Environment variables:
     * - CATOPY_LOG_LEVEL: DEBUG, INFO, WARN, ERROR, CRITICAL, OFF
     * - CATOPY_LOG_FILE: Path to log file (enables file logging if set)
     * - CATOPY_LOG_CONSOLE: true/false (default: true)
     */
    void initialize_from_environment();

    /**
     * @brief Initialize logger for development environment
     * - Level: DEBUG
     * - Console: enabled
     * - File: disabled
     */
    void initialize_for_development();

    /**
     * @brief Initialize logger for production environment
     * - Level: INFO
     * - Console: enabled (only WARN+)
     * - File: enabled (all levels)
     */
    void initialize_for_production();

    /**
     * @brief Initialize logger for CUDA debugging
     * - Level: DEBUG
     * - Console: enabled
     * - File: enabled (for CUDA operations)
     * - Special CUDA operation logging
     */
    void initialize_for_cuda_debug();

    /**
     * @brief Initialize logger for testing environment
     * - Level: WARN (minimal output)
     * - Console: enabled
     * - File: disabled
     */
    void initialize_for_testing();

    /**
     * @brief Get log level from string
     * @param level_str String representation of log level
     * @return Logger::Level enum value
     */
    Logger::Level string_to_level(const std::string& level_str);

    /**
     * @brief Get current environment type
     * @return String describing current environment
     */
    std::string get_environment_type();

    /**
     * @brief Check if running in debug mode
     * @return true if debug mode, false otherwise
     */
    bool is_debug_mode();

    /**
     * @brief Check if running in production mode
     * @return true if production mode, false otherwise
     */
    bool is_production_mode();

} // namespace LoggingConfig

#endif // LOGGING_CONFIG_CUH
