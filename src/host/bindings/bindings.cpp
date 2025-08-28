#include <pybind11/pybind11.h>
#include "devices_bindings.hpp"
#include "data_structures_bindings.hpp"
#include "../../../include/logger/Logger.cuh"

PYBIND11_MODULE(cato, m) {
    m.doc() = "Cato CUDA module";

    // Initialize logger for the entire module (por defecto OFF)
    Logger::initialize(Logger::Level::DEBUG, false);

    //* Agregar control del logging al módulo Python
    m.def("logger", [](bool enable) {
        Logger::enable_logging(enable);
        return enable;
    }, pybind11::arg("enable") = false, 
       "Enable or disable logging globally. True for DEBUG mode, False for OFF mode (default).");

    // Bind device detection functionality
    bind_devices(m);

    // Bind data structures (vectors, matrices, tensors)
    bind_data_structures(m);
}
