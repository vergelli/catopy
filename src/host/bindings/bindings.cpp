#include <pybind11/pybind11.h>
#include "devices_bindings.hpp"
#include "data_structures_bindings.hpp"

PYBIND11_MODULE(cato, m) {
    m.doc() = "Cato CUDA module";

    // Bind device detection functionality
    bind_devices(m);

    // Bind data structures (vectors, matrices, tensors)
    bind_data_structures(m);
}
