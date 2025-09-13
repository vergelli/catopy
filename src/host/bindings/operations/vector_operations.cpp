// src/host/bindings/operations/vector_operations.cpp
#include "vector_operations.hpp"
#include "../../../include/device/wrappers/VectorOperationWrapper.cuh"
#include "../../../include/device/devices.cuh"
#include "../../../include/core/caVector.cuh"
#include "../../../include/logger/Logger.cuh"

namespace py = pybind11;

namespace cato {
namespace bindings {

/**
 * @brief Bind vector operations to Python module
 * 
 * Creates Python bindings for vector operations using pybind11. This function
 * exposes both internal wrapper classes and public API functions to Python,
 * enabling high-performance vector operations from Python code.
 * 
 * @param m Python module to bind operations to
 * 
 * @note Creates internal VectorOperationWrapper class for advanced usage
 * @note Exposes public API functions for common vector operations
 * @note Uses static wrapper instances for performance optimization
 * @note Automatically selects first available CUDA device
 * @note Includes comprehensive error handling for device availability
 */
void bind_vector_operations(pybind11::module_& m) {
    Logger::debug("Binding vector operations module");

    // VectorOperationWrapper (para uso interno, no público)
    py::class_<VectorOperationWrapper>(m, "_VectorOperationWrapper")
        .def(py::init<const Device&>())
        .def("vecmul", &VectorOperationWrapper::vecmul)
        .def("vecadd", &VectorOperationWrapper::vecadd)
        .def("vecsub", &VectorOperationWrapper::vecsub)
        .def("get_device", &VectorOperationWrapper::get_device);

    // Funciones globales de operaciones (interfaz pública)
    m.def("vecmul", [](const caVector<double>& a, const caVector<double>& b) {
        Logger::debug("Python vecmul called with vectors of size {} and {}", a.size(), b.size());

        // Crear wrapper con el primer dispositivo disponible
        static Devices devices;
        if (devices.count() == 0) {
            throw std::runtime_error("No CUDA devices available");
        }

        static VectorOperationWrapper wrapper(devices.get_device(0));
        return wrapper.vecmul(a, b);
    }, py::arg("a"), py::arg("b"), 
       "Element-wise multiplication of two vectors: result[i] = a[i] * b[i]");

    m.def("vecadd", [](const caVector<double>& a, const caVector<double>& b) {
        Logger::debug("Python vecadd called with vectors of size {} and {}", a.size(), b.size());

        static Devices devices;
        if (devices.count() == 0) {
            throw std::runtime_error("No CUDA devices available");
        }

        static VectorOperationWrapper wrapper(devices.get_device(0));
        return wrapper.vecadd(a, b);
    }, py::arg("a"), py::arg("b"), 
       "Element-wise addition of two vectors: result[i] = a[i] + b[i]");

    m.def("vecsub", [](const caVector<double>& a, const caVector<double>& b) {
        Logger::debug("Python vecsub called with vectors of size {} and {}", a.size(), b.size());

        static Devices devices;
        if (devices.count() == 0) {
            throw std::runtime_error("No CUDA devices available");
        }

        static VectorOperationWrapper wrapper(devices.get_device(0));
        return wrapper.vecsub(a, b);
    }, py::arg("a"), py::arg("b"), 
       "Element-wise subtraction of two vectors: result[i] = a[i] - b[i]");
    
    Logger::debug("Vector operations module bound successfully");
}

// Implementación de las funciones globales

/**
 * @brief Template specialization for vector multiplication
 * 
 * C++ template specialization for element-wise vector multiplication.
 * Provides a direct C++ interface for vector operations without Python overhead.
 * 
 * @param a First input vector
 * @param b Second input vector
 * @return Result vector containing element-wise multiplication
 * @throws std::runtime_error if no CUDA devices are available
 * 
 * @note Uses static wrapper instance for performance optimization
 * @note Automatically selects first available CUDA device
 * @note Provides C++ interface for vector operations
 * @note Delegates to VectorOperationWrapper for actual computation
 */
template<>
caVector<double> vecmul<caVector<double>>(const caVector<double>& a, const caVector<double>& b) {
    static Devices devices;
    if (devices.count() == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    static VectorOperationWrapper wrapper(devices.get_device(0));
    return wrapper.vecmul(a, b);
}

/**
 * @brief Template specialization for vector addition
 * 
 * C++ template specialization for element-wise vector addition.
 * Provides a direct C++ interface for vector operations without Python overhead.
 * 
 * @param a First input vector
 * @param b Second input vector
 * @return Result vector containing element-wise addition
 * @throws std::runtime_error if no CUDA devices are available
 * 
 * @note Uses static wrapper instance for performance optimization
 * @note Automatically selects first available CUDA device
 * @note Provides C++ interface for vector operations
 * @note Delegates to VectorOperationWrapper for actual computation
 */
template<>
caVector<double> vecadd<caVector<double>>(const caVector<double>& a, const caVector<double>& b) {
    static Devices devices;
    if (devices.count() == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    static VectorOperationWrapper wrapper(devices.get_device(0));
    return wrapper.vecadd(a, b);
}

/**
 * @brief Template specialization for vector subtraction
 * 
 * C++ template specialization for element-wise vector subtraction.
 * Provides a direct C++ interface for vector operations without Python overhead.
 * 
 * @param a First input vector
 * @param b Second input vector
 * @return Result vector containing element-wise subtraction
 * @throws std::runtime_error if no CUDA devices are available
 * 
 * @note Uses static wrapper instance for performance optimization
 * @note Automatically selects first available CUDA device
 * @note Provides C++ interface for vector operations
 * @note Delegates to VectorOperationWrapper for actual computation
 */
template<>
caVector<double> vecsub<caVector<double>>(const caVector<double>& a, const caVector<double>& b) {
    static Devices devices;
    if (devices.count() == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    static VectorOperationWrapper wrapper(devices.get_device(0));
    return wrapper.vecsub(a, b);
}

// Scalar operations implementations

/**
 * @brief Template specialization for scalar vector multiplication
 * 
 * C++ template specialization for scalar multiplication on vectors.
 * Multiplies each element of the vector by a scalar value.
 * 
 * @param a Input vector to multiply
 * @param scalar Scalar value to multiply with each element
 * @return Result vector containing scalar multiplication
 * @throws std::runtime_error if no CUDA devices are available
 * 
 * @note Uses static wrapper instance for performance optimization
 * @note Automatically selects first available CUDA device
 * @note Provides C++ interface for scalar vector operations
 * @note Delegates to VectorOperationWrapper for actual computation
 */
template<>
caVector<double> vecmul_scalar<caVector<double>>(const caVector<double>& a, double scalar) {
    static Devices devices;
    if (devices.count() == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    static VectorOperationWrapper wrapper(devices.get_device(0));
    return wrapper.vecmul_scalar(a, scalar);
}

/**
 * @brief Template specialization for scalar vector addition
 * 
 * C++ template specialization for scalar addition on vectors.
 * Adds a scalar value to each element of the vector.
 * 
 * @param a Input vector to add scalar to
 * @param scalar Scalar value to add to each element
 * @return Result vector containing scalar addition
 * @throws std::runtime_error if no CUDA devices are available
 * 
 * @note Uses static wrapper instance for performance optimization
 * @note Automatically selects first available CUDA device
 * @note Provides C++ interface for scalar vector operations
 * @note Delegates to VectorOperationWrapper for actual computation
 */
template<>
caVector<double> vecadd_scalar<caVector<double>>(const caVector<double>& a, double scalar) {
    static Devices devices;
    if (devices.count() == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    static VectorOperationWrapper wrapper(devices.get_device(0));
    return wrapper.vecadd_scalar(a, scalar);
}

} // namespace bindings
} // namespace cato
