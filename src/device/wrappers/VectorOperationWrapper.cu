// src/device/wrappers/VectorOperationWrapper.cu
#include "../../../include/device/wrappers/VectorOperationWrapper.cuh"
#include "../../../include/device/kernels/vector_kernels.cuh"
#include "../../../include/device/DataStructureInfo.cuh" // Para VectorInfo
#include <chrono>

namespace cato {

/**
 * @brief Constructs a VectorOperationWrapper for the specified device
 * 
 * Initializes the vector operation wrapper with device information and creates
 * a vector-specific launch optimizer. This wrapper provides high-level vector
 * operations with automatic kernel launch optimization.
 * 
 * @param device CUDA device information for vector operations
 * 
 * @note Creates a VectorLaunchOptimizer instance for automatic optimization
 * @note Logs initialization for debugging purposes
 * @note All vector operations will be performed on the specified device
 * @note The wrapper manages memory transfers and kernel execution automatically
 */
VectorOperationWrapper::VectorOperationWrapper(const Device& device)
    : device_(device)
    , optimizer_(std::make_unique<VectorLaunchOptimizer>(device)) {

    Logger::debug("VectorOperationWrapper initialized for device: {}", device.get_name());
}

/**
 * @brief Perform element-wise vector multiplication
 * 
 * Multiplies two vectors element by element: result[i] = a[i] * b[i].
 * Uses optimized kernel launch configuration and automatic memory management.
 * 
 * @param a First input vector
 * @param b Second input vector
 * @return Result vector containing element-wise multiplication
 * 
 * @note Vectors must have the same size
 * @note Operation is performed on GPU with automatic optimization
 * @note Result is automatically transferred back to CPU
 * @note Uses vecmul_kernel for actual computation
 * @see execute_elementwise for the underlying implementation
 */
caVector<double> VectorOperationWrapper::vecmul(const caVector<double>& a, const caVector<double>& b) {
    return execute_elementwise(a, b, 
        [](const double* a_ptr, const double* b_ptr, double* result_ptr, size_t n, dim3 grid, dim3 block, cudaStream_t stream) {
            cato::kernels::vecmul_kernel<<<grid, block, 0, stream>>>(a_ptr, b_ptr, result_ptr, n);
        }, 
        "vecmul");
}

/**
 * @brief Perform element-wise vector addition
 * 
 * Adds two vectors element by element: result[i] = a[i] + b[i].
 * Uses optimized kernel launch configuration and automatic memory management.
 * 
 * @param a First input vector
 * @param b Second input vector
 * @return Result vector containing element-wise addition
 * 
 * @note Vectors must have the same size
 * @note Operation is performed on GPU with automatic optimization
 * @note Result is automatically transferred back to CPU
 * @note Uses vecadd_kernel for actual computation
 * @see execute_elementwise for the underlying implementation
 */
caVector<double> VectorOperationWrapper::vecadd(const caVector<double>& a, const caVector<double>& b) {
    return execute_elementwise(a, b,
        [](const double* a_ptr, const double* b_ptr, double* result_ptr, size_t n, dim3 grid, dim3 block, cudaStream_t stream) {
            cato::kernels::vecadd_kernel<<<grid, block, 0, stream>>>(a_ptr, b_ptr, result_ptr, n);
        },
        "vecadd");
}

/**
 * @brief Perform element-wise vector subtraction
 * 
 * Subtracts two vectors element by element: result[i] = a[i] - b[i].
 * Uses optimized kernel launch configuration and automatic memory management.
 * 
 * @param a First input vector
 * @param b Second input vector
 * @return Result vector containing element-wise subtraction
 * 
 * @note Vectors must have the same size
 * @note Operation is performed on GPU with automatic optimization
 * @note Result is automatically transferred back to CPU
 * @note Uses vecsub_kernel for actual computation
 * @see execute_elementwise for the underlying implementation
 */
caVector<double> VectorOperationWrapper::vecsub(const caVector<double>& a, const caVector<double>& b) {
    return execute_elementwise(a, b,
        [](const double* a_ptr, const double* b_ptr, double* result_ptr, size_t n, dim3 grid, dim3 block, cudaStream_t stream) {
            cato::kernels::vecsub_kernel<<<grid, block, 0, stream>>>(a_ptr, b_ptr, result_ptr, n);
        },
        "vecsub");
}

/**
 * @brief Perform scalar multiplication on vector
 * 
 * Multiplies each element of the vector by a scalar value: result[i] = a[i] * scalar.
 * Uses optimized kernel launch configuration and includes performance timing.
 * 
 * @param a Input vector to multiply
 * @param scalar Scalar value to multiply with each element
 * @return Result vector containing scalar multiplication
 * 
 * @note Operation is performed on GPU with automatic optimization
 * @note Result is automatically transferred back to CPU
 * @note Uses vecmul_scalar_kernel for actual computation
 * @note Includes performance timing and debug logging
 * @note Forces GPU to CPU transfer for Python compatibility
 */
caVector<double> VectorOperationWrapper::vecmul_scalar(const caVector<double>& a, double scalar) {
    auto start_time = std::chrono::high_resolution_clock::now();

    Logger::debug("Starting vecmul_scalar operation on vector of size {} with scalar {}", a.size(), scalar);

    caVector<double> result = create_result_vector(a);

    caVector<double> a_gpu = a;
    a_gpu.ensure_on_gpu();
    result.ensure_on_gpu();

    // Obtener información del vector para optimización
    VectorInfo vector_info(a.size(), sizeof(double));

    // Optimizar configuración de lanzamiento
    LaunchConfig config = optimizer_->optimize_for_operation(vector_info, "vecmul_scalar");

    // Obtener punteros GPU
    double* a_ptr = a_gpu.gpu_ptr();
    double* result_ptr = result.gpu_ptr();

    // Lanzar kernel de multiplicación por escalar
    cato::kernels::vecmul_scalar_kernel<<<config.gridDim, config.blockDim, 0, config.stream>>>(
        a_ptr, scalar, result_ptr, a.size());

    // Sincronizar
    cudaDeviceSynchronize();

    // Marcar resultado como dirty en GPU
    result.mark_gpu_dirty();

    // CRITICAL: Forzar transferencia de GPU a CPU
    result.ensure_on_host();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double time_ms = duration.count() / 1000.0;

    Logger::debug("vecmul_scalar operation completed in {:.3f} ms", time_ms);

    return result;
}

/**
 * @brief Perform scalar addition on vector
 * 
 * Adds a scalar value to each element of the vector: result[i] = a[i] + scalar.
 * Uses optimized kernel launch configuration and includes performance timing.
 * 
 * @param a Input vector to add scalar to
 * @param scalar Scalar value to add to each element
 * @return Result vector containing scalar addition
 * 
 * @note Operation is performed on GPU with automatic optimization
 * @note Result is automatically transferred back to CPU
 * @note Uses vecadd_scalar_kernel for actual computation
 * @note Includes performance timing and debug logging
 * @note Forces GPU to CPU transfer for Python compatibility
 */
caVector<double> VectorOperationWrapper::vecadd_scalar(const caVector<double>& a, double scalar) {
    auto start_time = std::chrono::high_resolution_clock::now();

    Logger::debug("Starting vecadd_scalar operation on vector of size {} with scalar {}", a.size(), scalar);

    // Crear vector resultado
    caVector<double> result = create_result_vector(a);

    // Asegurar que el vector esté en GPU
    caVector<double> a_gpu = a;
    a_gpu.ensure_on_gpu();
    result.ensure_on_gpu();

    // Obtener información del vector para optimización
    VectorInfo vector_info(a.size(), sizeof(double));

    // Optimizar configuración de lanzamiento
    LaunchConfig config = optimizer_->optimize_for_operation(vector_info, "vecadd_scalar");

    // Obtener punteros GPU
    double* a_ptr = a_gpu.gpu_ptr();
    double* result_ptr = result.gpu_ptr();

    // Lanzar kernel de suma con escalar
    cato::kernels::vecadd_scalar_kernel<<<config.gridDim, config.blockDim, 0, config.stream>>>(
        a_ptr, scalar, result_ptr, a.size());

    // Sincronizar
    cudaDeviceSynchronize();

    // Marcar resultado como dirty en GPU
    result.mark_gpu_dirty();

    // CRITICAL: Forzar transferencia de GPU a CPU
    result.ensure_on_host();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double time_ms = duration.count() / 1000.0;

    Logger::debug("vecadd_scalar operation completed in {:.3f} ms", time_ms);

    return result;
}

/**
 * @brief Execute element-wise operation with optimized kernel launch
 * 
 * Generic template function that executes element-wise operations on two vectors
 * using optimized kernel launch configuration. Handles memory management,
 * validation, and performance timing automatically.
 * 
 * @tparam KernelFunc Type of kernel function to execute
 * @param a First input vector
 * @param b Second input vector
 * @param kernel_func Kernel function to execute (lambda or function pointer)
 * @param operation_name Name of the operation for logging purposes
 * @return Result vector containing the operation result
 * 
 * @note Validates vector compatibility before execution
 * @note Automatically manages GPU memory transfers
 * @note Uses optimized launch configuration from VectorLaunchOptimizer
 * @note Includes debug logging and performance timing
 * @note Forces GPU to CPU transfer for Python compatibility
 * @note Includes debug verification of GPU results
 */
template<typename KernelFunc>
caVector<double> VectorOperationWrapper::execute_elementwise(const caVector<double>& a, 
                                                           const caVector<double>& b,
                                                           KernelFunc kernel_func,
                                                           const std::string& operation_name) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Validar compatibilidad de vectores
    validate_vector_compatibility(a, b);

    log_operation_start(operation_name, a.size());

    // Crear vector resultado
    caVector<double> result = create_result_vector(a);

    // Asegurar que los vectores estén en GPU
    caVector<double> a_gpu = a;
    caVector<double> b_gpu = b;
    ensure_vectors_on_gpu(a_gpu, b_gpu);
    ensure_vectors_on_gpu(result, result);

    // Obtener información del vector para optimización
    VectorInfo vector_info(a.size(), sizeof(double));

    // Optimizar configuración de lanzamiento
    LaunchConfig config = optimizer_->optimize_for_operation(vector_info, operation_name);

    // Obtener punteros GPU
    double* a_ptr = a_gpu.gpu_ptr();
    double* b_ptr = b_gpu.gpu_ptr();
    double* result_ptr = result.gpu_ptr();

    // Lanzar kernel con configuración optimizada
    kernel_func(a_ptr, b_ptr, result_ptr, a.size(), config.gridDim, config.blockDim, config.stream);

    // Sincronizar
    cudaDeviceSynchronize();

    // Marcar resultado como dirty en GPU
    result.mark_gpu_dirty();
    
    // Debug: Verificar resultado en GPU
    double gpu_result[5];
    cudaMemcpy(gpu_result, result_ptr, 5 * sizeof(double), cudaMemcpyDeviceToHost);
    Logger::debug("GPU result after kernel: [{}, {}, {}, {}, {}]", 
                  gpu_result[0], gpu_result[1], gpu_result[2], gpu_result[3], gpu_result[4]);
    
    // CRITICAL: Forzar transferencia de GPU a CPU para que Python pueda leer los datos
    result.ensure_on_host();
    Logger::debug("Forced GPU to CPU transfer completed");

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double time_ms = duration.count() / 1000.0;

    log_operation_complete(operation_name, time_ms);

    return result;
}

/**
 * @brief Validate compatibility between two vectors for element-wise operations
 * 
 * Checks that both vectors have the same size and are not empty before
 * performing element-wise operations. Throws exceptions for invalid conditions.
 * 
 * @param a First vector to validate
 * @param b Second vector to validate
 * @throws std::invalid_argument if vectors have different sizes
 * @throws std::invalid_argument if either vector is empty
 * 
 * @note Called before all element-wise operations
 * @note Ensures vectors are compatible for element-wise processing
 * @note Provides detailed error messages for debugging
 * @note Prevents runtime errors from size mismatches
 */
void VectorOperationWrapper::validate_vector_compatibility(const caVector<double>& a, const caVector<double>& b) const {
    if (a.size() != b.size()) {
        throw std::invalid_argument("VectorOperationWrapper: Vector sizes must match (" + 
                                  std::to_string(a.size()) + " vs " + std::to_string(b.size()) + ")");
    }
    
    if (a.size() == 0) {
        throw std::invalid_argument("VectorOperationWrapper: Vectors cannot be empty");
    }
}

/**
 * @brief Create a result vector with the same size as input vector
 * 
 * Creates a new caVector with the same size as the input vector and initializes
 * all elements to zero. This is used to prepare the result vector for operations.
 * 
 * @param a Input vector to match size with
 * @return New vector with same size as input, initialized to zeros
 * 
 * @note Creates vector with identical size to input vector
 * @note Initializes all elements to 0.0
 * @note Uses lambda function for initialization
 * @note Used by all vector operations to create result containers
 */
caVector<double> VectorOperationWrapper::create_result_vector(const caVector<double>& a) const {
    // Crear vector resultado con el mismo tamaño que 'a'
    return caVector<double>(a.size(), [](double* data, size_t size, const std::vector<double>& params) {
        // Inicializar con ceros
        for (size_t i = 0; i < size; ++i) {
            data[i] = 0.0;
        }
    });
}

/**
 * @brief Ensure both vectors are available on GPU
 * 
 * Forces both input vectors to be transferred to GPU memory if they are not
 * already present. This ensures that kernel operations can access the data.
 * 
 * @param a First vector to ensure on GPU
 * @param b Second vector to ensure on GPU
 * 
 * @note Calls ensure_on_gpu() on both vectors
 * @note Used before kernel execution to guarantee GPU availability
 * @note Handles memory transfers automatically if needed
 * @note Essential for proper kernel execution
 */
void VectorOperationWrapper::ensure_vectors_on_gpu(caVector<double>& a, caVector<double>& b) const {
    a.ensure_on_gpu();
    b.ensure_on_gpu();
}

/**
 * @brief Log the start of a vector operation
 * 
 * Outputs debug information about the beginning of a vector operation,
 * including the operation name and vector size for debugging and monitoring.
 * 
 * @param operation Name of the operation being started
 * @param vector_size Size of the vector being processed
 * 
 * @note Only outputs when debug logging is enabled
 * @note Provides context for operation monitoring
 * @note Used for debugging operation performance and behavior
 * @note Helps track operation execution flow
 */
void VectorOperationWrapper::log_operation_start(const std::string& operation, size_t vector_size) const {
    Logger::debug("Starting {} operation on vector of size {}", operation, vector_size);
}

/**
 * @brief Log the completion of a vector operation
 * 
 * Outputs debug information about the completion of a vector operation,
 * including the operation name and execution time for performance monitoring.
 * 
 * @param operation Name of the operation that completed
 * @param time_ms Execution time in milliseconds
 * 
 * @note Only outputs when debug logging is enabled
 * @note Provides performance metrics for operation monitoring
 * @note Used for debugging operation efficiency and timing
 * @note Helps track operation performance characteristics
 */
void VectorOperationWrapper::log_operation_complete(const std::string& operation, double time_ms) const {
    Logger::debug("{} operation completed in {:.3f} ms", operation, time_ms);
}

} // namespace cato
