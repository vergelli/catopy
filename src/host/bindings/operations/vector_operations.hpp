// src/host/bindings/operations/vector_operations.hpp
#ifndef VECTOR_OPERATIONS_HPP
#define VECTOR_OPERATIONS_HPP

#include <pybind11/pybind11.h>

namespace cato {
namespace bindings {

/**
 * @brief Bindings for vector operations
 * 
 * This module provides Python bindings for vector operations like
 * multiplication, addition, and subtraction using the VectorOperationWrapper.
 * 
 * The operations are exposed as both global functions and methods
 * that can be used with caVector objects.
 */
void bind_vector_operations(pybind11::module_& m);

// Forward declarations for global functions
template<typename T>
T vecmul(const T& a, const T& b);
template<typename T>
T vecadd(const T& a, const T& b);
template<typename T>
T vecsub(const T& a, const T& b);

// Scalar operations
template<typename T>
T vecmul_scalar(const T& a, double scalar);
template<typename T>
T vecadd_scalar(const T& a, double scalar);

} // namespace bindings
} // namespace cato

#endif // VECTOR_OPERATIONS_HPP
