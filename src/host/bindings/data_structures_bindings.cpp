#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../../../include/core/caVector.cuh"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream> // Added for debug prints

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
        .def("ensure_on_gpu", &caVector<double>::ensure_on_gpu)
        .def("ensure_on_host", &caVector<double>::ensure_on_host)
        .def("print_info", &caVector<double>::print_info)
        .def("get_memory_info", &caVector<double>::get_memory_info)
        .def("get_init_params", &caVector<double>::get_init_params)
        .def("__len__", &caVector<double>::size)
        .def("__str__", [](const caVector<double>& vec) {
            return vec.smart_string();
        })
        .def("__repr__", [](const caVector<double>& vec) {
            return vec.smart_string();
        })
        .def("to_list_string", &caVector<double>::to_list_string)
        .def("head_string", &caVector<double>::head_string)
        .def("tail_string", &caVector<double>::tail_string)
        .def("smart_string", &caVector<double>::smart_string);

    //& ===== INITIALIZATION FUNCTIONS =====
    //& These return std::function objects that can be used directly

    m.def("zeros", []() { 
        std::cout << "DEBUG: Python zeros() function called" << std::endl;

        //& Return a Python function that returns a list of zeros
        return py::cpp_function([](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            std::cout << "DEBUG: Python zeros function called with size: " << size << std::endl;

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                result.append(0.0);
            }

            std::cout << "DEBUG: Python zeros function completed successfully" << std::endl;
            return result;
        });
    }, "Create initialization function for zeros");

    m.def("ones", []() { 
        std::cout << "DEBUG: Python ones() function called" << std::endl;

        //& Return a Python function that returns a list of ones
        return py::cpp_function([](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            std::cout << "DEBUG: Python ones function called with size: " << size << std::endl;

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                result.append(1.0);
            }

            std::cout << "DEBUG: Python ones function completed successfully" << std::endl;
            return result;
        });
    }, "Create initialization function for ones");

    m.def("constant", [](double value) { 
        std::cout << "DEBUG: Python constant(" << value << ") function called" << std::endl;

        //& Return a Python function that returns a list of constant values
        //& This function will be called directly by ca.vector()
        return py::cpp_function([value](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            std::cout << "DEBUG: Python constant function called with size: " << size << ", value: " << value << std::endl;

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                result.append(value);
                std::cout << "DEBUG: Appending value " << value << " to result[" << i << "]" << std::endl;
            }

            std::cout << "DEBUG: Python constant function completed successfully" << std::endl;
            return result;
        });
    }, py::arg("value"), "Create initialization function for constant value");

    //& Random functions
    m.def("random", [](int seed = -1) { 
        std::cout << "DEBUG: Python random(" << seed << ") function called" << std::endl;

        //& Return a Python function that returns a list of random values
        return py::cpp_function([seed](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            std::cout << "DEBUG: Python random function called with size: " << size << ", seed: " << seed << std::endl;

            // Simple random implementation
            std::srand(seed >= 0 ? seed : std::time(nullptr));

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                double random_value = static_cast<double>(std::rand()) / RAND_MAX;
                result.append(random_value);
                std::cout << "DEBUG: Appending random value " << random_value << " to result[" << i << "]" << std::endl;
            }

            std::cout << "DEBUG: Python random function completed successfully" << std::endl;
            return result;
        });
    }, py::arg("seed") = -1, "Create initialization function for random values");

    m.def("uniform", [](double min = 0.0, double max = 1.0, int seed = -1) { 
        std::cout << "DEBUG: Python uniform(" << min << ", " << max << ", " << seed << ") function called" << std::endl;

        //& Return a Python function that returns a list of uniform random values
        return py::cpp_function([min, max, seed](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            std::cout << "DEBUG: Python uniform function called with size: " << size << ", min: " << min << ", max: " << max << ", seed: " << seed << std::endl;

            std::srand(seed >= 0 ? seed : std::time(nullptr));
            double range = max - min;

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                double uniform_value = min + (static_cast<double>(std::rand()) / RAND_MAX) * range;
                result.append(uniform_value);
                std::cout << "DEBUG: Appending uniform value " << uniform_value << " to result[" << i << "]" << std::endl;
            }

            std::cout << "DEBUG: Python uniform function completed successfully" << std::endl;
            return result;
        });
    }, py::arg("min") = 0.0, py::arg("max") = 1.0, py::arg("seed") = -1,
       "Create initialization function for uniform distribution");

    m.def("normal", [](double mean = 0.0, double std = 1.0, int seed = -1) { 
        std::cout << "DEBUG: Python normal(" << mean << ", " << std << ", " << seed << ") function called" << std::endl;

        //& Return a Python function that returns a list of normal random values
        return py::cpp_function([mean, std, seed](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            std::cout << "DEBUG: Python normal function called with size: " << size << ", mean: " << mean << ", std: " << std << ", seed: " << seed << std::endl;

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
                std::cout << "DEBUG: Appending normal value " << normal_value << " to result[" << i << "]" << std::endl;
            }

            std::cout << "DEBUG: Python normal function completed successfully" << std::endl;
            return result;
        });
    }, py::arg("mean") = 0.0, py::arg("std") = 1.0, py::arg("seed") = -1,
       "Create initialization function for normal distribution");

    m.def("box_muller", [](double mean = 0.0, double std = 1.0, int seed = -1) { 
        std::cout << "DEBUG: Python box_muller(" << mean << ", " << std << ", " << seed << ") function called" << std::endl;

        //& Return a Python function that returns a list of Box-Muller random values
        return py::cpp_function([mean, std, seed](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            std::cout << "DEBUG: Python box_muller function called with size: " << size << ", mean: " << mean << ", std: " << std << ", seed: " << seed << std::endl;

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
                std::cout << "DEBUG: Appending Box-Muller value " << value0 << " to result[" << i << "]" << std::endl;

                if (i + 1 < size) {
                    double value1 = mean + std * z1;
                    result.append(value1);
                    std::cout << "DEBUG: Appending Box-Muller value " << value1 << " to result[" << i + 1 << "]" << std::endl;
                }
            }

            std::cout << "DEBUG: Python box_muller function completed successfully" << std::endl;
            return result;
        });
    }, py::arg("mean") = 0.0, py::arg("std") = 1.0, py::arg("seed") = -1,
       "Create initialization function for Box-Muller distribution");

    m.def("sequence", [](double start = 0.0, double step = 1.0) { 
        std::cout << "DEBUG: Python sequence(" << start << ", " << step << ") function called" << std::endl;

        //& Return a Python function that returns a list of sequence values
        return py::cpp_function([start, step](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            std::cout << "DEBUG: Python sequence function called with size: " << size << ", start: " << start << ", step: " << step << std::endl;

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                double sequence_value = start + i * step;
                result.append(sequence_value);
                std::cout << "DEBUG: Appending sequence value " << sequence_value << " to result[" << i << "]" << std::endl;
            }

            std::cout << "DEBUG: Python sequence function completed successfully" << std::endl;
            return result;
        });
    }, py::arg("start") = 0.0, py::arg("step") = 1.0,
       "Create initialization function for arithmetic sequence");

    m.def("arange", [](double start = 0.0, double stop = 1.0, double step = 1.0) { 
        std::cout << "DEBUG: Python arange(" << start << ", " << stop << ", " << step << ") function called" << std::endl;

        //& Return a Python function that returns a list of arange values
        return py::cpp_function([start, stop, step](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            std::cout << "DEBUG: Python arange function called with size: " << size << ", start: " << start << ", stop: " << stop << ", step: " << step << std::endl;

            py::list result;

            for (size_t i = 0; i < size; ++i) {
                double current = start + (i * step);
                result.append(current);
                std::cout << "DEBUG: Appending arange value " << current << " to result[" << i << "]" << std::endl;
            }

            std::cout << "DEBUG: Python arange function completed successfully" << std::endl;
            return result;
        });
    }, py::arg("start") = 0.0, py::arg("stop") = 1.0, py::arg("step") = 1.0,
       "Create initialization function for arange-like sequence");

    m.def("mathematical", [](const std::string& func_name = "sin") { 
        std::cout << "DEBUG: Python mathematical('" << func_name << "') function called" << std::endl;

        //& Return a Python function that returns a list of mathematical function values
        return py::cpp_function([func_name](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; //! Unused parameter
            std::cout << "DEBUG: Python mathematical function called with size: " << size << ", func_name: " << func_name << std::endl;

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
                std::cout << "DEBUG: Appending mathematical value " << math_value << " (func=" << func_name << ", x=" << x << ") to result[" << i << "]" << std::endl;
            }

            std::cout << "DEBUG: Python mathematical function completed successfully" << std::endl;
            return result;
        });
    }, py::arg("func_name") = "sin",
       "Create initialization function for mathematical functions");

    m.def("sine", [](double frequency = 1.0, double amplitude = 1.0, double phase = 0.0) { 
        std::cout << "DEBUG: Python sine(" << frequency << ", " << amplitude << ", " << phase << ") function called" << std::endl;

        //& Return a Python function that returns a list of sine wave values
        return py::cpp_function([frequency, amplitude, phase](size_t size, const std::vector<double>& params) -> py::list {
            (void)params; // Unused parameter
            std::cout << "DEBUG: Python sine function called with size: " << size << ", frequency: " << frequency << ", amplitude: " << amplitude << ", phase: " << phase << std::endl;

            py::list result;
            for (size_t i = 0; i < size; ++i) {
                double x = static_cast<double>(i) * 0.1;
                double sine_value = amplitude * std::sin(2.0 * M_PI * frequency * x + phase);

                result.append(sine_value);
                std::cout << "DEBUG: Appending sine value " << sine_value << " (f=" << frequency << ", A=" << amplitude << ", φ=" << phase << ", x=" << x << ") to result[" << i << "]" << std::endl;
            }

            std::cout << "DEBUG: Python sine function completed successfully" << std::endl;
            return result;
        });
    }, py::arg("frequency") = 1.0, py::arg("amplitude") = 1.0, py::arg("phase") = 0.0,
       "Create initialization function for sine wave");

    //& ===== CONVENIENCE FUNCTIONS =====

    m.def("vector", [](size_t size, py::object init_func_obj, 
                       const std::vector<double>& params = std::vector<double>{}) {
        std::cout << "DEBUG: Python vector() function called with size=" << size << std::endl;
        std::cout << "DEBUG: init_func_obj type: " << py::str(init_func_obj.get_type()) << std::endl;
        std::cout << "DEBUG: params size: " << params.size() << std::endl;

        try {
            std::cout << "DEBUG: About to check if init_func_obj is callable..." << std::endl;

            py::object result;
            //& Check if the object is callable and handle accordingly
            if (py::hasattr(init_func_obj, "__call__")) {
                //& Check if this is a function that expects parameters (like normal(), box_muller(), etc.)
                //& These functions expect (size, params) but we're passing (size, params) directly
                std::cout << "DEBUG: init_func_obj is callable, checking if it's a parameterized function..." << std::endl;

                //& Try to call it directly first (for functions like normal(), box_muller(), etc.)
                try {
                    result = init_func_obj;
                    std::cout << "DEBUG: Using init_func_obj directly as parameterized function" << std::endl;
                } catch (...) {
                    //& If that fails, try calling it without parameters
                    std::cout << "DEBUG: Direct call failed, trying to call without parameters" << std::endl;
                    result = init_func_obj();
                }
            } else {
                std::cout << "DEBUG: init_func_obj is not callable, calling it without parameters" << std::endl;
                result = init_func_obj();
            }

            std::cout << "DEBUG: Result object type: " << py::str(result.get_type()) << std::endl;

            std::cout << "DEBUG: About to create caVector..." << std::endl;

            //& Create the vector directly using the result object
            //& This avoids std::function conversion completely
            std::cout << "DEBUG: Creating caVector directly..." << std::endl;

            //& Use a lambda that calls the Python function and copies data back
            auto init_lambda = [result](double* data, size_t size, const std::vector<double>& params) {
                std::cout << "DEBUG: Direct lambda called with data=" << data << ", size=" << size << std::endl;

                //& Call the Python function directly
                py::gil_scoped_acquire gil;  //! Ensure Python GIL is held
                py::object py_func = result;

                //& Call Python function to get the data
                py::list py_data = py_func(py::cast(size), py::cast(params));

                //& Copy data from Python list to C++ array
                for (size_t i = 0; i < size; ++i) {
                    data[i] = py::cast<double>(py_data[i]);
                    std::cout << "DEBUG: Copied data[" << i << "] = " << data[i] << std::endl;
                }

                std::cout << "DEBUG: Direct lambda completed successfully" << std::endl;
            };

            caVector<double> vec(size, init_lambda, params);
            std::cout << "DEBUG: caVector created successfully" << std::endl;

            std::cout << "DEBUG: About to cast to Python..." << std::endl;
            auto result_obj = py::cast(vec);
            std::cout << "DEBUG: Cast to Python completed successfully" << std::endl;
            return result_obj;

        } catch (const std::exception& e) {
            std::cout << "DEBUG: ERROR - Exception during vector creation: " << e.what() << std::endl;
            throw std::runtime_error("Failed to create vector: " + std::string(e.what()));
        } catch (...) {
            std::cout << "DEBUG: ERROR - Unknown exception during vector creation" << std::endl;
            throw std::runtime_error("Failed to create vector: unknown error");
        }
    }, py::arg("size"), py::arg("init_func"), py::arg("params") = std::vector<double>{},
       "Create a vector with the specified size and initialization function");
}
