#include "../../include/logger/LoggingConfig.cuh"
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include "../logger/Logger.cuh"

namespace LoggingConfig {

void initialize_from_environment() {
    //& Get log level from environment
    const char* level_env = std::getenv("CATOPY_LOG_LEVEL");
    Logger::Level level = Logger::Level::DEBUG; //* Default to DEBUG

    if (level_env) {
        level = string_to_level(level_env);
    }

    // Check if file logging is enabled
    const char* file_env = std::getenv("CATOPY_LOG_FILE");
    bool log_to_file = (file_env != nullptr);
    std::string log_file_path = log_to_file ? file_env : "logs/catopy.log";

    // Check if console logging is enabled
    const char* console_env = std::getenv("CATOPY_LOG_CONSOLE");
    bool log_to_console = true; // Default to true
    if (console_env) {
        std::string console_str = console_env;
        std::transform(console_str.begin(), console_str.end(), console_str.begin(), ::tolower);
        log_to_console = (console_str == "true" || console_str == "1" || console_str == "yes");
    }

    // Initialize logger
    if (log_to_console) {
        Logger::initialize(level, log_to_file, log_file_path);
    } else if (log_to_file) {
        // Only file logging
        Logger::initialize(level, true, log_file_path);
    } else {
        // Fallback to console if both are disabled
        Logger::initialize(level, false);
    }
}

void initialize_for_development() {
    Logger::initialize(Logger::Level::DEBUG, false);
}

void initialize_for_production() {
    Logger::initialize(Logger::Level::DEBUG, true, "logs/catopy_prod.log");
}

void initialize_for_cuda_debug() {
    Logger::initialize(Logger::Level::DEBUG, true, "logs/catopy_cuda.log");
}

void initialize_for_testing() {
    Logger::initialize(Logger::Level::DEBUG, false);
}

Logger::Level string_to_level(const std::string& level_str) {
    std::string upper_level = level_str;
    std::transform(upper_level.begin(), upper_level.end(), upper_level.begin(), ::toupper);
    
    if (upper_level == "DEBUG") return Logger::Level::DEBUG;
    if (upper_level == "OFF") return Logger::Level::OFF;

    //& Default to DEBUG if unknown
    return Logger::Level::DEBUG;
}

std::string get_environment_type() {
    const char* env_type = std::getenv("CATOPY_ENV");
    if (env_type) {
        return env_type;
    }

    if (std::getenv("DEBUG") || std::getenv("DEVELOPMENT")) {
        return "development";
    }
    if (std::getenv("PRODUCTION") || std::getenv("PROD")) {
        return "production";
    }
    if (std::getenv("TESTING") || std::getenv("TEST")) {
        return "testing";
    }

    return "development";
}

bool is_debug_mode() {
    std::string env = get_environment_type();
    return (env == "development" || env == "debug");
}

bool is_production_mode() {
    std::string env = get_environment_type();
    return (env == "production" || env == "prod");
}

} // namespace LoggingConfig
