#ifndef LOGGER_CUH
#define LOGGER_CUH

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/fmt/fmt.h>
#include <memory>
#include <string>
#include <iostream>
#include <filesystem>

/**
 * @file Logger.cuh
 * @brief Centralized logging system for the catopy project
 * 
 * This logger provides:
 * - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
 * - Console and file output
 * - CUDA-friendly async logging
 * - Configurable log levels per environment
 * - Thread-safe logging
 */

class Logger {
private:
    static std::shared_ptr<spdlog::logger> logger_;
    static bool initialized_;
    static bool logging_enabled_;  // Control global del logging
    
    // Private constructor to prevent instantiation
    Logger() = delete;

public:
    // Log levels - Solo DEBUG y OFF para librería numérica
    enum class Level {
        DEBUG = spdlog::level::debug,
        OFF = spdlog::level::off
    };

    /**
     * @brief Initialize the logger with default configuration
     * @param level Minimum log level to display
     * @param log_to_file Whether to also log to file
     * @param log_file_path Path to log file (if logging to file)
     */
    static void initialize(Level level = Level::DEBUG, 
                          bool log_to_file = false, 
                          const std::string& log_file_path = "logs/catopy.log");

    /**
     * @brief Set the minimum log level
     * @param level New minimum log level
     */
    static void set_level(Level level);

    /**
     * @brief Get current log level
     * @return Current log level
     */
    static Level get_level();

    /**
     * @brief Enable or disable logging globally
     * @param enable true for DEBUG mode, false for OFF mode
     * 
     * This method is the main control point for the entire library.
     * When called, it affects ALL modules and components.
     */
    static void enable_logging(bool enable);

    /**
     * @brief Check if logger is initialized
     * @return true if initialized, false otherwise
     */
    static bool is_initialized();

    /**
     * @brief Check if logging is currently enabled
     * @return true if DEBUG mode, false if OFF mode
     */
    static bool is_logging_enabled();

    /**
     * @brief Shutdown the logger
     */
    static void shutdown();

    // ===== LOGGING METHODS =====

    /**
     * @brief Log a debug message (only when logging is enabled)
     */
    template<typename... Args>
    static void debug(const char* fmt, const Args&... args) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->debug(fmt, args...);
        }
    }

    /**
     * @brief Log an info message (only when logging is enabled)
     */
    template<typename... Args>
    static void info(const char* fmt, const Args&... args) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->info(fmt, args...);
        }
    }

    /**
     * @brief Log a warning message (only when logging is enabled)
     */
    template<typename... Args>
    static void warn(const char* fmt, const Args&... args) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->warn(fmt, args...);
        }
    }

    /**
     * @brief Log an error message (only when logging is enabled)
     */
    template<typename... Args>
    static void error(const char* fmt, const Args&... args) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->error(fmt, args...);
        }
    }

    /**
     * @brief Log a critical message (only when logging is enabled)
     */
    template<typename... Args>
    static void critical(const char* fmt, const Args&... args) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->critical(fmt, args...);
        }
    }

    // ===== CONVENIENCE METHODS =====

    /**
     * @brief Log CUDA operation start
     */
    static void cuda_start(const char* operation) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->debug("🚀 CUDA {} START", operation);
        }
    }

    /**
     * @brief Log CUDA operation completion
     */
    static void cuda_complete(const char* operation) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->debug("CUDA {} COMPLETE", operation);
        }
    }

    /**
     * @brief Log CUDA operation error
     */
    static void cuda_error(const char* operation, const char* error_msg) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->error("CUDA {} ERROR: {}", operation, error_msg);
        }
    }

    /**
     * @brief Log memory allocation
     */
    static void memory_alloc(const char* type, size_t size) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->debug("MEMORY ALLOC {}: {} bytes", type, size);
        }
    }

    /**
     * @brief Log memory deallocation
     */
    static void memory_free(const char* type, size_t size) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->debug("MEMORY FREE {}: {} bytes", type, size);
        }
    }

    /**
     * @brief Log memory transfer
     */
    static void memory_transfer(const char* from, const char* to, size_t size) {
        if (is_initialized() && is_logging_enabled()) {
            logger_->debug("MEMORY TRANSFER {} → {}: {} bytes", from, to, size);
        }
    }

    // ===== UTILITY METHODS =====

    /**
     * @brief Flush all pending log messages
     */
    static void flush();

    /**
     * @brief Get logger instance (for advanced usage)
     * @return Shared pointer to the logger
     */
    static std::shared_ptr<spdlog::logger> get_logger();
};

// ===== INLINE IMPLEMENTATIONS =====

inline std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;
inline bool Logger::initialized_ = false;
inline bool Logger::logging_enabled_ = false;  // Por defecto OFF

inline void Logger::initialize(Level level, bool log_to_file, const std::string& log_file_path) {
    if (initialized_) {
        return;
    }

    try {
        std::vector<spdlog::sink_ptr> sinks;
        
        // Console sink with colors
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] [%t] %v");
        sinks.push_back(console_sink);

        // File sink if requested
        if (log_to_file) {
            try {
                // Create logs directory if it doesn't exist
                std::filesystem::create_directories(std::filesystem::path(log_file_path).parent_path());
                
                auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file_path, 0, 0);
                file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] [%s:%#] %v");
                sinks.push_back(file_sink);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not create file sink: " << e.what() << std::endl;
            }
        }

        // Create logger with multiple sinks
        logger_ = std::make_shared<spdlog::logger>("catopy", sinks.begin(), sinks.end());
        
        // Set log level
        logger_->set_level(static_cast<spdlog::level::level_enum>(level));
        
        // Set as default logger
        spdlog::set_default_logger(logger_);
        
        initialized_ = true;
        
        // Por defecto, logging está deshabilitado (modo silencioso)
        logging_enabled_ = false;
        
        // Solo log de inicialización si está habilitado
        if (logging_enabled_) {
            logger_->info("🚀 Logger initialized successfully (level: {}, file logging: {})", 
                         spdlog::level::to_string_view(static_cast<spdlog::level::level_enum>(level)),
                         log_to_file ? "enabled" : "disabled");
        }
                     
    } catch (const std::exception& e) {
        std::cerr << "Error initializing logger: " << e.what() << std::endl;
        // Fallback to basic console logging
        logger_ = spdlog::stdout_color_mt("catopy");
        logger_->set_level(static_cast<spdlog::level::level_enum>(level));
        initialized_ = true;
    }
}

inline void Logger::set_level(Level level) {
    if (is_initialized()) {
        logger_->set_level(static_cast<spdlog::level::level_enum>(level));
        logger_->info("Log level changed to: {}", spdlog::level::to_string_view(static_cast<spdlog::level::level_enum>(level)));
    }
}

inline Logger::Level Logger::get_level() {
    if (is_initialized()) {
        return static_cast<Level>(logger_->level());
    }
    return Level::DEBUG;
}

inline bool Logger::is_initialized() {
    return initialized_ && logger_ != nullptr;
}

inline void Logger::shutdown() {
    if (initialized_) {
        if (logger_) {
            logger_->info("Logger shutting down");
            logger_->flush();
        }
        spdlog::shutdown();
        initialized_ = false;
    }
}

inline void Logger::flush() {
    if (is_initialized()) {
        logger_->flush();
    }
}

inline std::shared_ptr<spdlog::logger> Logger::get_logger() {
    return logger_;
}

// ===== NUEVOS MÉTODOS DE CONTROL GLOBAL =====

inline void Logger::enable_logging(bool enable) {
    if (is_initialized()) {
        logging_enabled_ = enable;
        
        if (enable) {
            // Modo DEBUG: mostrar todos los logs
            logger_->set_level(spdlog::level::debug);
            logger_->info(" DEBUG MODE ENABLED - All logging is now visible");
        } else {
            // Modo OFF: silenciar todos los logs
            logger_->set_level(spdlog::level::off);
            logger_->info("LOGGING DISABLED - All logging is now silent");
        }
    }
}

inline bool Logger::is_logging_enabled() {
    return logging_enabled_;
}

#endif // LOGGER_CUH
