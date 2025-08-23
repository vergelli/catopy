#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // permite usar std::vector y std::unordered_map
#include "devices.cuh"
#include "devices_bindings.hpp"


namespace py = pybind11;

void bind_devices(py::module_& m) {
    py::class_<Devices>(m, "Devices")
        .def(py::init<>())
        .def("get_devices", &Devices::get_devices)
        .def("print_devices", &Devices::print_devices)
        .def("get_properties", &Devices::get_properties);
}
