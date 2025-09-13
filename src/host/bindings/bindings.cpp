#include <pybind11/pybind11.h>
#include <filesystem>
#include "devices_bindings.hpp"
#include "data_structures_bindings.hpp"
#include "operations/vector_operations.hpp"
#include "../../../include/logger/Logger.cuh"

PYBIND11_MODULE(cato, m) {
    m.doc() = "Cato CUDA module";

    // TODO: WIP - Helper function for cross-platform log paths (needs testing)
    // Helper function to get standard log directory
    auto get_standard_log_path = []() -> std::string {
        std::filesystem::path home_dir;

        #ifdef _WIN32
            // Windows: Use APPDATA environment variable
            const char* appdata = std::getenv("APPDATA");
            if (appdata) {
                home_dir = std::filesystem::path(appdata);
            }
        #elif defined(__APPLE__)
            // macOS: Use HOME environment variable
            const char* home = std::getenv("HOME");
            if (home) {
                home_dir = std::filesystem::path(home) / "Library" / "Logs";
            }
        #else
            // Linux and other Unix-like systems: Use HOME environment variable
            const char* home = std::getenv("HOME");
            if (home) {
                home_dir = std::filesystem::path(home) / ".local" / "share";
            }
        #endif

        // If we couldn't determine home directory, fallback to current directory
        if (home_dir.empty()) {
            home_dir = std::filesystem::current_path();
        }

        // Append catopy/logs subdirectory
        return (home_dir / "catopy" / "logs").string();
    };

    // Initialize logger for the entire module (console only by default)
    Logger::initialize(Logger::Level::DEBUG, false);

    m.def("logger", [&get_standard_log_path](bool enable, bool log_to_file = false, const std::string& log_file_path = "") {
        // Si se especifica logging a archivo, reconfigurar el logger
        if (log_to_file) {
            std::string actual_log_path = log_file_path.empty() ? 
                get_standard_log_path() + "/catopy.log" : log_file_path;

            // TODO: WIP - File logging functionality needs debugging
            // Reinitialize logger with file logging
            Logger::initialize(Logger::Level::DEBUG, true, actual_log_path);

            // Después de reconfigurar, habilitar logging
            Logger::enable_logging(enable);
        } else {
            // Solo habilitar/deshabilitar logging sin cambiar configuración de archivo
            Logger::enable_logging(enable);
        }

        return enable;
    }, pybind11::arg("enable") = false, 
       pybind11::arg("log_to_file") = false,
       pybind11::arg("log_file_path") = "",
       "Enable or disable logging globally. True for DEBUG mode, False for OFF mode (default).\n"
       "If log_to_file is True, logs will be written to file. Use log_file_path to specify custom path.");

    bind_devices(m);
    bind_data_structures(m);

    cato::bindings::bind_vector_operations(m);
}
