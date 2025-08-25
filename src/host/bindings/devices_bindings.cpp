#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // permite usar std::vector y std::unordered_map
#include "devices.cuh"
#include "devices_bindings.hpp"

namespace py = pybind11;

void bind_devices(py::module_& m) {
    // Bind de la clase Device
    py::class_<Device>(m, "Device")
        .def("get_id", &Device::get_id)
        .def("get_name", &Device::get_name)
        .def("get_compute_capability", &Device::get_compute_capability)
        .def("get_total_global_memory", &Device::get_total_global_memory)
        .def("get_shared_memory_per_block", &Device::get_shared_memory_per_block)
        .def("get_registers_per_block", &Device::get_registers_per_block)
        .def("get_warp_size", &Device::get_warp_size)
        .def("get_max_threads_per_block", &Device::get_max_threads_per_block)
        .def("get_max_threads_per_multiprocessor", &Device::get_max_threads_per_multiprocessor)
        .def("get_multiprocessor_count", &Device::get_multiprocessor_count)
        .def("get_memory_bus_width", &Device::get_memory_bus_width)
        .def("get_l2_cache_size", &Device::get_l2_cache_size)
        .def("get_max_grid_size", &Device::get_max_grid_size)
        .def("get_max_threads_dim", &Device::get_max_threads_dim)
        .def("get_total_constant_memory", &Device::get_total_constant_memory)
        .def("show", &Device::show)
        .def("get_properties", &Device::get_properties)
        .def("__str__", [](const Device& d) {
            auto props = d.get_properties();
            std::string result = "Device(id=" + std::to_string(d.get_id()) + ", name='" + d.get_name() + "')\n";
            result += "Properties:\n";
            for (const auto& [key, value] : props) {
                result += "  " + key + ": " + value + "\n";
            }
            return result;
        })
        .def("__repr__", [](const Device& d) {
            return "Device(id=" + std::to_string(d.get_id()) + ", name='" + d.get_name() + "')";
        });

    // Bind de la clase Devices
    py::class_<Devices>(m, "Devices")
        .def(py::init<>())
        .def("count", &Devices::count)
        .def("get_device", &Devices::get_device)
        .def("show", &Devices::show)
        .def("__getitem__", &Devices::operator[])
        .def("__len__", &Devices::count)
        .def("__str__", [](const Devices& d) {
            return "Devices(count=" + std::to_string(d.count()) + ")";
        });
}
