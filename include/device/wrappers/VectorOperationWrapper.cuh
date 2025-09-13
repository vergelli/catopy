// include/device/wrappers/VectorOperationWrapper.cuh
#ifndef VECTOR_OPERATION_WRAPPER_CUH
#define VECTOR_OPERATION_WRAPPER_CUH

#include "../optimizers/VectorLaunchOptimizer.cuh"
#include "../../core/caVector.cuh"
#include <memory>
#include <string>

namespace cato {

/**
 * @brief Wrapper for vector operations with automatic optimization
 * 
 * This class acts as an intermediary between high-level operations
 * and optimized CUDA kernels, managing launch configuration and kernel execution.
 * It provides a unified interface for vector operations while automatically
 * optimizing kernel launch parameters for maximum performance.
 * 
 * @note All operations are performed on the GPU device specified during construction.
 * @note The wrapper automatically handles memory transfers and kernel launch optimization.
 * @note Vector operations require compatible vector sizes and proper device allocation.
 */
class VectorOperationWrapper {
private:
    std::unique_ptr<VectorLaunchOptimizer> optimizer_;
    Device device_;

public:
    explicit VectorOperationWrapper(const Device& device);
    ~VectorOperationWrapper() = default;

    caVector<double> vecmul(const caVector<double>& a, const caVector<double>& b);
    caVector<double> vecadd(const caVector<double>& a, const caVector<double>& b);
    caVector<double> vecsub(const caVector<double>& a, const caVector<double>& b);

    caVector<double> vecmul_scalar(const caVector<double>& a, double scalar);
    caVector<double> vecadd_scalar(const caVector<double>& a, double scalar);

    template<typename KernelFunc>
    caVector<double> execute_elementwise(const caVector<double>& a, 
                                       const caVector<double>& b,
                                       KernelFunc kernel_func,
                                       const std::string& operation_name = "elementwise");

    // Getters
    const Device& get_device() const { return device_; }
    const VectorLaunchOptimizer& get_optimizer() const { return *optimizer_; }

private:
    void validate_vector_compatibility(const caVector<double>& a, const caVector<double>& b) const;
    caVector<double> create_result_vector(const caVector<double>& a) const;
    void ensure_vectors_on_gpu(caVector<double>& a, caVector<double>& b) const;

    void log_operation_start(const std::string& operation, size_t vector_size) const;
    void log_operation_complete(const std::string& operation, double time_ms) const;
};

} // namespace cato

#endif // VECTOR_OPERATION_WRAPPER_CUH
