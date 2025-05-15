#include <pybind11/pybind11.h>
#include "devices_bindings.hpp"

PYBIND11_MODULE(cato, m) {
    m.doc() = "Cato CUDA module";
    bind_devices(m);
}
