#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../../../include/core/caVector.cuh"
#include "operations/vector_operations.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "../logger/Logger.cuh"

namespace py = pybind11;

void bind_data_structures(py::module_& m) {
    //& ===== CAVECTOR BINDINGS =====

    py::class_<caVector<double>>(m, "caVector")
        .def(py::init<size_t, std::function<void(double*, size_t, const std::vector<double>&)>, const std::vector<double>&>(),
             py::arg("size"),
             py::arg("init_func"),
             py::arg("params") = std::vector<double>{})
        .def("size", &caVector<double>::size)
        .def("is_on_gpu", &caVector<double>::is_on_gpu)
        .def("is_host_dirty", &caVector<double>::is_host_dirty)
        .def("is_gpu_dirty", &caVector<double>::is_gpu_dirty)
        .def("ensure_on_gpu", &caVector<double>::ensure_on_gpu)
        .def("ensure_on_host", &caVector<double>::ensure_on_host)
        .def("print_info", &caVector<double>::print_info)
        .def("get_memory_info", &caVector<double>::get_memory_info)
        .def("get_init_params", &caVector<double>::get_init_params)
        .def("copy", &caVector<double>::copy, "Create a deep copy of the vector")
        .def("__len__", &caVector<double>::size)
        .def("__getitem__", [](const caVector<double>& vec, size_t index) {
            return vec[index];
        }, py::arg("index"), "Get element at specified index")
        .def("__getitem__", [](const caVector<double>& vec, py::slice slice) {
            // Parse Python slice to get start, stop, step
            py::ssize_t start, stop, step, slice_length;
            if (!slice.compute(vec.size(), &start, &stop, &step, &slice_length)) {
                throw py::error_already_set();
            }

            Logger::debug("Python slice: start={} stop={} step={} slice_length={}", start, stop, step, slice_length);

            // Special case: v[::-1] - detect when Python passes v[4:4:-1] instead of v[4:-1:-1]
            if (step < 0 && start == vec.size() - 1 && stop == start) {
                // This is likely v[::-1], force stop to -1
                stop = -1;
                Logger::debug("Detected v[::-1] pattern, forcing stop=-1");
            }

            Logger::debug("Final slice parameters: start={} stop={} step={}", start, stop, step);

            // Call "our" slice method xD
            return vec.slice(start, stop, step);
        }, py::arg("slice"), "Get slice of vector using Python slice syntax")
        .def("__setitem__", [](caVector<double>& vec, size_t index, double value) {
            vec[index] = value;
        }, py::arg("index"), py::arg("value"), "Set element at specified index")
        .def("__str__", [](const caVector<double>& vec) {
            return vec.smart_string();
        })
        .def("__repr__", [](const caVector<double>& vec) {
            return vec.smart_string();
        })
        .def("to_list_string", &caVector<double>::to_list_string)
        .def("head_string", &caVector<double>::head_string)
        .def("tail_string", &caVector<double>::tail_string)
        .def("smart_string", &caVector<double>::smart_string)
        .def("at", static_cast<double&(caVector<double>::*)(size_t)>(&caVector<double>::at), py::arg("index"), "Get element at index with bounds checking")
        // Vector operations operators
        .def("__mul__", [](const caVector<double>& a, const caVector<double>& b) {
            Logger::debug("Python __mul__ called with vectors of size {} and {}", a.size(), b.size());
            return cato::bindings::vecmul(a, b);
        }, py::arg("other"), "Element-wise multiplication with another vector")
        .def("__add__", [](const caVector<double>& a, const caVector<double>& b) {
            Logger::debug("Python __add__ called with vectors of size {} and {}", a.size(), b.size());
            return cato::bindings::vecadd(a, b);
        }, py::arg("other"), "Element-wise addition with another vector")
        .def("__sub__", [](const caVector<double>& a, const caVector<double>& b) {
            Logger::debug("Python __sub__ called with vectors of size {} and {}", a.size(), b.size());
            return cato::bindings::vecsub(a, b);
        }, py::arg("other"), "Element-wise subtraction with another vector")
        // Scalar operations
        .def("__mul__", [](const caVector<double>& a, double scalar) {
            Logger::debug("Python __mul__ called with vector of size {} and scalar {}", a.size(), scalar);
            return cato::bindings::vecmul_scalar(a, scalar);
        }, py::arg("scalar"), "Element-wise multiplication with scalar")
        .def("__add__", [](const caVector<double>& a, double scalar) {
            Logger::debug("Python __add__ called with vector of size {} and scalar {}", a.size(), scalar);
            return cato::bindings::vecadd_scalar(a, scalar);
        }, py::arg("scalar"), "Element-wise addition with scalar")
        // Reverse operations (scalar * vector)
        .def("__rmul__", [](const caVector<double>& a, double scalar) {
            Logger::debug("Python __rmul__ called with scalar {} and vector of size {}", scalar, a.size());
            return cato::bindings::vecmul_scalar(a, scalar);
        }, py::arg("scalar"), "Element-wise multiplication with scalar (reverse)")
        .def("__radd__", [](const caVector<double>& a, double scalar) {
            Logger::debug("Python __radd__ called with scalar {} and vector of size {}", scalar, a.size());
            return cato::bindings::vecadd_scalar(a, scalar);
        }, py::arg("scalar"), "Element-wise addition with scalar (reverse)");

    //& ===== INITIALIZATION FUNCTIONS =====
    //& These return std::function objects that can be used directly

    m.def("zeros", []() { 
        Logger::debug("Python zeros() function called");

        //& Return a Python function that returns a list of zeros
        return py::cpp_function([](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            Logger::debug("Python zeros function called with size:  {}", size);

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                result.append(0.0);
            }

            Logger::debug("Python zeros function completed successfully");
            return result;
        });
    }, "Create initialization function for zeros");

    m.def("ones", []() { 
        Logger::debug("Python ones() function called");

        //& Return a Python function that returns a list of ones
        return py::cpp_function([](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            Logger::debug("Python ones function called with size:  {}", size);

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                result.append(1.0);
            }

            Logger::debug("Python ones function completed successfully");
            return result;
        });
    }, "Create initialization function for ones");

    m.def("constant", [](double value) { 
        Logger::debug("Python constant({}) function called", value);

        //& Return a Python function that returns a list of constant values
        //& This function will be called directly by ca.vector()
        return py::cpp_function([value](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            Logger::debug("Python constant function called with size: {}, value: {}", size, value);

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                result.append(value);
                Logger::debug("Appending value {} to result[{}]", value, i);
            }

            Logger::debug("Python constant function completed successfully");
            return result;
        });
    }, py::arg("value"), "Create initialization function for constant value");

    //& Random functions
    m.def("random", [](int seed = -1) { 
        Logger::debug("Python random({}) function called", seed);

        //& Return a Python function that returns a list of random values
        return py::cpp_function([seed](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            Logger::debug("Python random function called with size: {}, seed: {}", size, seed);

            // Simple random implementation
            std::srand(seed >= 0 ? seed : std::time(nullptr));

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                double random_value = static_cast<double>(std::rand()) / RAND_MAX;
                result.append(random_value);
                Logger::debug("Appending random value {} to result[{}]", random_value, i);
            }

            Logger::debug("Python random function completed successfully");
            return result;
        });
    }, py::arg("seed") = -1, "Create initialization function for random values");

    m.def("uniform", [](double min = 0.0, double max = 1.0, int seed = -1) { 
        Logger::debug("Python uniform({}, {}, {}) function called", min, max, seed);

        //& Return a Python function that returns a list of uniform random values
        return py::cpp_function([min, max, seed](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            Logger::debug("Python uniform function called with size: {}, min: {}, max: {}, seed: {}", size, min, max, seed);

            std::srand(seed >= 0 ? seed : std::time(nullptr));
            double range = max - min;

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                double uniform_value = min + (static_cast<double>(std::rand()) / RAND_MAX) * range;
                result.append(uniform_value);
                Logger::debug("Appending uniform value {} to result[{}]", uniform_value, i);
            }

            Logger::debug("Python uniform function completed successfully");
            return result;
        });
    }, py::arg("min") = 0.0, py::arg("max") = 1.0, py::arg("seed") = -1,
       "Create initialization function for uniform distribution");

    m.def("normal", [](double mean = 0.0, double std = 1.0, int seed = -1) { 
        Logger::debug("Python normal({}, {}, {}) function called", mean, std, seed);

        //& Return a Python function that returns a list of normal random values
        return py::cpp_function([mean, std, seed](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            Logger::debug("Python normal function called with size: {}, mean: {}, std: {}, seed: {}", size, mean, std, seed);

            std::srand(seed >= 0 ? seed : std::time(nullptr));

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                // Simple Box-Muller approximation
                double u1 = static_cast<double>(std::rand()) / RAND_MAX;
                double u2 = static_cast<double>(std::rand()) / RAND_MAX;
                if (u1 <= 0.0) u1 = 1e-10; // Avoid log(0)
                double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
                double normal_value = mean + std * z;

                result.append(normal_value);
                Logger::debug("Appending normal value {} to result[{}]", normal_value, i);
            }

            Logger::debug("Python normal function completed successfully");
            return result;
        });
    }, py::arg("mean") = 0.0, py::arg("std") = 1.0, py::arg("seed") = -1,
       "Create initialization function for normal distribution");

    m.def("box_muller", [](double mean = 0.0, double std = 1.0, int seed = -1) { 
        Logger::debug("Python box_muller({}, {}, {}) function called", mean, std, seed);

        //& Return a Python function that returns a list of Box-Muller random values
        return py::cpp_function([mean, std, seed](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            Logger::debug("Python box_muller function called with size: {}, mean: {}, std: {}, seed: {}", size, mean, std, seed);

            std::srand(seed >= 0 ? seed : std::time(nullptr));

            py::list result;
            for (size_t i = 0; i < size; i += 2) {
                double u1 = static_cast<double>(std::rand()) / RAND_MAX;
                double u2 = static_cast<double>(std::rand()) / RAND_MAX;
                if (u1 <= 0.0) u1 = 1e-10; // Avoid log(0)

                double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
                double z1 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);

                double value0 = mean + std * z0;
                result.append(value0);
                Logger::debug("Appending Box-Muller value {} to result[{}]", value0, i);

                if (i + 1 < size) {
                    double value1 = mean + std * z1;
                    result.append(value1);
                    Logger::debug("Appending Box-Muller value {} to result[{}]", value1, i + 1);
                }
            }

            Logger::debug("Python box_muller function completed successfully");
            return result;
        });
    }, py::arg("mean") = 0.0, py::arg("std") = 1.0, py::arg("seed") = -1,
       "Create initialization function for Box-Muller distribution");

    m.def("sequence", [](double start = 0.0, double step = 1.0) { 
        Logger::debug("Python sequence({}, {}) function called", start, step);

        //& Return a Python function that returns a list of sequence values
        return py::cpp_function([start, step](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            Logger::debug("Python sequence function called with size: {}, start: {}, step: {}", size, start, step);

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                double sequence_value = start + i * step;
                result.append(sequence_value);
                Logger::debug("Appending sequence value {} to result[{}]", sequence_value, i);
            }

            Logger::debug("Python sequence function completed successfully");
            return result;
        });
    }, py::arg("start") = 0.0, py::arg("step") = 1.0,
       "Create initialization function for arithmetic sequence");

    m.def("arange", [](double start = 0.0, double stop = 1.0, double step = 1.0) { 
        Logger::debug("Python arange({}, {}, {}) function called", start, stop, step);

        //& Return a Python function that returns a list of arange values
        return py::cpp_function([start, stop, step](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            Logger::debug("Python arange function called with size: {}, start: {}, stop: {}, step: {}", size, start, stop, step);

            py::list result;

            for (size_t i = 0; i < size; ++i) {
                double current = start + (i * step);
                result.append(current);
                Logger::debug("Appending arange value {} to result[{}]", current, i);
            }

            Logger::debug("Python arange function completed successfully");
            return result;
        });
    }, py::arg("start") = 0.0, py::arg("stop") = 1.0, py::arg("step") = 1.0,
       "Create initialization function for arange-like sequence");

    m.def("mathematical", [](const std::string& func_name = "sin") { 
        Logger::debug("Python mathematical('{}') function called", func_name);

        //& Return a Python function that returns a list of mathematical function values
        return py::cpp_function([func_name](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; //! Unused parameter
            Logger::debug("Python mathematical function called with size: {}, func_name: {}", size, func_name);

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                double x = static_cast<double>(i) * 0.1; //? Scale for better visualization
                double math_value;

                if (func_name == "sin") {
                    math_value = std::sin(x);
                } else if (func_name == "cos") {
                    math_value = std::cos(x);
                } else if (func_name == "tan") {
                    math_value = std::tan(x);
                } else if (func_name == "exp") {
                    math_value = std::exp(-x); //? Negative for bounded values
                } else if (func_name == "log") {
                    math_value = std::log(1.0 + x); //? +1 to avoid log(0)
                } else {
                    math_value = x; //? Default to linear
                }

                result.append(math_value);
                Logger::debug("Appending mathematical value {} (func={}, x={}) to result[{}]", math_value, func_name, x, i);
            }

            Logger::debug("Python mathematical function completed successfully");
            return result;
        });
    }, py::arg("func_name") = "sin",
       "Create initialization function for mathematical functions");

    m.def("sine", [](double frequency = 1.0, double amplitude = 1.0, double phase = 0.0) { 
        Logger::debug("Python sine({}, {}, {}) function called", frequency, amplitude, phase);

        //& Return a Python function that returns a list of sine wave values
        return py::cpp_function([frequency, amplitude, phase](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            Logger::debug("Python sine function called with size: {}, frequency: {}, amplitude: {}, phase: {}", size, frequency, amplitude, phase);

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                double x = static_cast<double>(i) * 0.1;
                double sine_value = amplitude * std::sin(2.0 * M_PI * frequency * x + phase);

                result.append(sine_value);
                Logger::debug("Appending sine value {} (f={}, A={}, φ={}, x={}) to result[{}]", sine_value, frequency, amplitude, phase, x, i);
            }

            Logger::debug("Python sine function completed successfully");
            return result;
        });
    }, py::arg("frequency") = 1.0, py::arg("amplitude") = 1.0, py::arg("phase") = 0.0,
       "Create initialization function for sine wave");

    //& ===== CONVENIENCE FUNCTIONS =====

    m.def("vector", [](size_t size, py::object init_func_obj, 
                       const std::vector<double>& params = std::vector<double>{}) {
        Logger::debug("Python vector() function called with size= {}", size);
        Logger::debug("init_func_obj type:  {}", py::str(init_func_obj.get_type()).cast<std::string>());
        Logger::debug("params size:  {}", params.size());

        try {
            Logger::debug("About to check if init_func_obj is callable...");

            py::object result;
            //& Check if the object is callable and handle accordingly
            if (py::hasattr(init_func_obj, "__call__")) {
                //& Check if this is a function that expects parameters (like normal(), box_muller(), etc.)
                //& These functions expect (size, params) but we're passing (size, params) directly
                Logger::debug("init_func_obj is callable, checking if it's a parameterized function...");

                //& Try to call it directly first (for functions like normal(), box_muller(), etc.)
                try {
                    result = init_func_obj;
                    Logger::debug("Using init_func_obj directly as parameterized function");
                } catch (...) {
                    //& If that fails, try calling it without parameters
                    Logger::debug("Direct call failed, trying to call without parameters");
                    result = init_func_obj();
                }
            } else {
                Logger::debug("init_func_obj is not callable, calling it without parameters");
                result = init_func_obj();
            }

            Logger::debug("Result object type:  {}", py::str(result.get_type()).cast<std::string>());

            Logger::debug("About to create caVector...");

            //& Create the vector directly using the result object
            //& This avoids std::function conversion completely
            Logger::debug("Creating caVector directly...");

            //& Use a lambda that calls the Python function and copies data back
            auto init_lambda = [result](double* data, size_t size, const std::vector<double>& params) {
                Logger::debug("Direct lambda called with data={}, size={}", static_cast<void*>(data), size);

                //& Call the Python function directly
                py::gil_scoped_acquire gil;  //! Ensure Python GIL is held
                py::object py_func = result;

                //& Call Python function to get the data
                py::list py_data = py_func(py::cast(size), py::cast(params));

                //& Copy data from Python list to C++ array
                for (size_t i = 0; i < size; ++i) {
                    data[i] = py::cast<double>(py_data[i]);
                    Logger::debug("Copied data[{}] = {}", i, data[i]);
                }

                Logger::debug("Direct lambda completed successfully");
            };

            caVector<double> vec(size, init_lambda, params);
            Logger::debug("caVector created successfully");

            Logger::debug("About to cast to Python...");
            auto result_obj = py::cast(vec);
            Logger::debug("Cast to Python completed successfully");
            return result_obj;

        } catch (const std::exception& e) {
            Logger::debug("ERROR - Exception during vector creation:  {}", e.what());
            throw std::runtime_error("Failed to create vector: " + std::string(e.what()));
        } catch (...) {
            Logger::debug("ERROR - Unknown exception during vector creation");
            throw std::runtime_error("Failed to create vector: unknown error");
        }
    }, py::arg("size"), py::arg("init_func"), py::arg("params") = std::vector<double>{},
       "Create a vector with the specified size and initialization function");
}
