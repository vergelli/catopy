#ifndef CA_VECTOR_CUH
#define CA_VECTOR_CUH

#include "../device/memory/GpuBuffer.cuh"
#include "../device/cuda_errors.cuh"
#include "../logger/Logger.cuh"
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <algorithm>

/**
 * @file caVector.cuh
 * @brief 1D Vector class with lazy copy memory management
 * 
 * This class implements a 1-dimensional vector with intelligent memory management
 * that follows the "lazy copy" pattern:
 * - Data is initially allocated and stored in HOST memory
 * - GPU memory is allocated only when needed for computation
 * - Automatic memory transfer between HOST and GPU as required
 lo siguiente
 * Key features:
 * - Lazy copy by default (memory stays on HOST until GPU is needed)
 * - Automatic GPU memory management when required
 * - Support for custom initialization functions
 * - Memory-efficient design
 * 
 * @tparam T Data type (e.g., double, float, int)
 */
template<typename T>
class caVector {
private:
    // Core data
    std::vector<T> host_data;                                   // Data stored in HOST memory
    std::unique_ptr<GpuBuffer<T>> gpu_buffer;                   // GPU memory (allocated on-demand)

    // Metadata
    size_t size_;                                               // Number of elements
    bool is_on_gpu_;                                            // Whether data is currently on GPU

    // Initialization function and parameters
    std::function<void(T*, size_t, const std::vector<double>&)> init_function;  // Function to initialize data
    std::vector<double> init_params;                            // Parameters for initialization function

public:
    /**
     * @brief Constructor: creates vector with specified size and initialization
     * @param size Number of elements in the vector
     * @param init_func Function to initialize vector elements
     * @param params Optional parameters for initialization function
     *
     * Example:
     * @code
     * // Create vector with 1000 zeros
     * caVector<double> vec(1000, [](double* data, size_t size, const std::vector<double>& params) {
     *     for (size_t i = 0; i < size; ++i) data[i] = 0.0;
     * });
     * @endcode
     */
    caVector(size_t size, 
             std::function<void(T*, size_t, const std::vector<double>&)> init_func,
             const std::vector<double>& params = {})
        : size_(size)
        , is_on_gpu_(false)
        , init_function(init_func)
        , init_params(params) {
        
        Logger::debug("caVector constructor called with size={}", size);
        Logger::debug("init_func is null? {}", (init_func ? "No" : "Yes"));
        Logger::debug("params size: {}", params.size());
        
        if (size == 0) {
            Logger::error("ERROR - size is 0");
            throw std::invalid_argument("caVector: size must be greater than 0");
        }

        if (!init_func) {
            Logger::error("ERROR - init_func is null");
            throw std::invalid_argument("caVector: initialization function cannot be null");
        }

        Logger::debug("About to resize host_data to {}", size);
        
        // Allocate HOST memory
        try {
            host_data.resize(size);
            Logger::debug("host_data resized successfully to  {}", host_data.size());
        } catch (const std::exception& e) {
            Logger::debug("ERROR - Failed to resize host_data:  {}", e.what());
            throw;
        }

        Logger::debug("host_data.data() is null?  {}", (host_data.data() == nullptr ? "Yes" : "No"));
        Logger::debug("host_data.size() == size?  {}", (host_data.size() == size ? "Yes" : "No"));
        
        // Initialize data using the provided function
        if (host_data.data() != nullptr && host_data.size() == size) {
                        Logger::debug("About to call init_function with data={}, size={}", static_cast<void*>(host_data.data()), size);
            Logger::debug("host_data.capacity() =  {}", host_data.capacity());
            Logger::debug("sizeof(T) =  {}", sizeof(T));
            Logger::debug("Total allocated bytes =  {}", (host_data.capacity() * sizeof(T)));
            
            // Verify memory boundaries
            Logger::debug("Memory range: {} to {}", static_cast<void*>(host_data.data()), static_cast<void*>(host_data.data() + size));
            Logger::debug("init_function address:  {}", static_cast<void*>(&init_function));
            
            try {
                Logger::debug("Calling init_function...");
                init_function(host_data.data(), size, init_params);
                Logger::debug("init_function completed successfully");
                
                // Verify data was written correctly
                Logger::debug("Verifying data after init_function...");
                for (size_t i = 0; i < std::min(size, size_t(5)); ++i) { // Check first 5 elements
                    Logger::debug("data[{}] = {}", i, host_data[i]);
                }
                
            } catch (const std::exception& e) {
                Logger::debug("ERROR - init_function threw exception:  {}", e.what());
                throw;
            }
        } else {
            Logger::debug("ERROR - Memory allocation failed or size mismatch");
            Logger::debug("host_data.data() =  {}", static_cast<void*>(host_data.data()));
            Logger::debug("host_data.size() =  {}", host_data.size());
            Logger::debug("expected size =  {}", size);
            throw std::runtime_error("caVector: memory allocation failed or size mismatch");
        }
        
        Logger::debug("caVector constructor completed successfully");
    }

    /**
     * @brief Template constructor: creates vector with direct initialization function
     * @param size Number of elements in the vector
     * @param init_func Direct initialization function (avoiding std::function conversion)
     * @param params Optional parameters for initialization function
     *
     * This constructor avoids the std::function conversion issues that cause stack smashing
     */
    template<typename InitFunc>
    caVector(size_t size, InitFunc&& init_func, const std::vector<double>& params = {})
        : size_(size)
        , is_on_gpu_(false)
        , init_params(params) {
        
        Logger::debug("caVector TEMPLATE constructor called with size= {}", size);
        Logger::debug("Using direct function call (no std::function conversion)");
        Logger::debug("params size:  {}", params.size());
        
        if (size == 0) {
            Logger::debug("ERROR - size is 0");
            throw std::invalid_argument("caVector: size must be greater than 0");
        }

        Logger::debug("About to resize host_data to  {}", size);

        // Allocate HOST memory
        try {
            host_data.resize(size);
            Logger::debug("host_data resized successfully to  {}", host_data.size());
        } catch (const std::exception& e) {
            Logger::debug("ERROR - Failed to resize host_data:  {}", e.what());
            throw;
        }

        Logger::debug("host_data.data() is null?  {}", (host_data.data() == nullptr ? "Yes" : "No"));
        Logger::debug("host_data.size() == size?  {}", (host_data.size() == size ? "Yes" : "No"));

        // Initialize data using the provided function directly
        if (host_data.data() != nullptr && host_data.size() == size) {
            Logger::debug("About to call init_func directly with data={}, size={}", static_cast<void*>(host_data.data()), size);
            Logger::debug("host_data.capacity() =  {}", host_data.capacity());
            Logger::debug("sizeof(T) =  {}", sizeof(T));
            Logger::debug("Total allocated bytes =  {}", (host_data.capacity() * sizeof(T)));

            // Verify memory boundaries
            Logger::debug("Memory range: {} to {}", static_cast<void*>(host_data.data()), static_cast<void*>(host_data.data() + size));

            try {
                Logger::debug("Calling init_func directly...");
                init_func(host_data.data(), size, params);
                Logger::debug("init_func completed successfully");

                // Verify data was written correctly
                Logger::debug("Verifying data after init_func...");
                for (size_t i = 0; i < std::min(size, size_t(5)); ++i) { // Check first 5 elements
                    Logger::debug("data[{}] = {}", i, host_data[i]);
                }

            } catch (const std::exception& e) {
                Logger::debug("ERROR - init_func threw exception:  {}", e.what());
                throw;
            }
        } else {
            Logger::debug("ERROR - Memory allocation failed or size mismatch");
            Logger::debug("host_data.data() =  {}", static_cast<void*>(host_data.data()));
            Logger::debug("host_data.size() =  {}", host_data.size());
            Logger::debug("expected size =  {}", size);
            throw std::runtime_error("caVector: memory allocation failed or size mismatch");
        }

        Logger::debug("caVector TEMPLATE constructor completed successfully");
    }

    /**
     * @brief Destructor: automatically cleans up GPU memory if allocated
     */
    ~caVector() {
        Logger::debug("caVector destructor called for size= {}", size_);
        Logger::debug("is_on_gpu_ =  {}", (is_on_gpu_ ? "true" : "false"));
        Logger::debug("gpu_buffer exists?  {}", (gpu_buffer ? "Yes" : "No"));
        Logger::debug("host_data.size() =  {}", host_data.size());
        Logger::debug("host_data.data() =  {}", static_cast<void*>(host_data.data()));
        
        try {
            // Clean up GPU memory if allocated
            if (gpu_buffer) {
                Logger::debug("About to reset gpu_buffer");
                gpu_buffer.reset();
                Logger::debug("gpu_buffer reset completed");
            } else {
                Logger::debug("No gpu_buffer to clean up");
            }
        } catch (const std::exception& e) {
            Logger::debug("ERROR - Exception in destructor:  {}", e.what());
            // Ignore exceptions in destructor
        } catch (...) {
            Logger::debug("ERROR - Unknown exception in destructor");
        }
        
        Logger::debug("About to exit caVector destructor");
        Logger::debug("caVector destructor completed");
    }

    // ===== MEMORY MANAGEMENT =====

    /**
     * @brief Ensures data is available on GPU
     * 
     * This method implements the "lazy copy" pattern:
     * - If data is already on GPU, does nothing
     * - If data is only on HOST, allocates GPU memory and transfers data
     * - Sets is_on_gpu_ flag to true
     */
    void ensure_on_gpu() {
        if (!is_on_gpu_) {
            // Allocate GPU memory
            gpu_buffer = std::make_unique<GpuBuffer<T>>(size_);

            // Transfer data from HOST to GPU
            CHECK_CUDA_ERROR(cudaMemcpy(
                gpu_buffer->get_pointer(),
                host_data.data(),
                size_ * sizeof(T),
                cudaMemcpyHostToDevice
            ));
            is_on_gpu_ = true;
        }
    }

    /**
     * @brief Ensures data is available on HOST
     * 
     * If data was modified on GPU, this method transfers it back to HOST
     * to ensure consistency.
     */
    void ensure_on_host() {
        if (is_on_gpu_) {
            // Transfer data from GPU to HOST
            CHECK_CUDA_ERROR(cudaMemcpy(
                host_data.data(),
                gpu_buffer->get_pointer(),
                size_ * sizeof(T),
                cudaMemcpyDeviceToHost
            ));
            // Update flag to indicate data is now on HOST
            is_on_gpu_ = false;
        }
    }

    // ===== DATA ACCESS =====

    /**
     * @brief Get pointer to HOST data
     * @return Raw pointer to HOST memory
     * 
     * Use this for CPU operations. Data is guaranteed to be on HOST.
     */
    T* host_ptr() {
        ensure_on_host();                                            // Ensure data is on HOST
        return host_data.data();
    }

    /**
     * @brief Get pointer to GPU data
     * @return Raw pointer to GPU memory
     * 
     * Use this for CUDA operations. Data is guaranteed to be on GPU.
     */
    T* gpu_ptr() {
        ensure_on_gpu();  // Ensure data is on GPU
        return gpu_buffer->get_pointer();
    }

    /**
     * @brief Get const reference to HOST data
     * @return Const reference to HOST data vector
     */
    const std::vector<T>& host_data_ref() const {
        return host_data;
    }

    // ===== METADATA =====

    /**
     * @brief Get vector size
     * @return Number of elements in the vector
     */
    size_t size() const { return size_; }

    /**
     * @brief Check if data is currently on GPU
     * @return true if data is on GPU, false if only on HOST
     */
    bool is_on_gpu() const { return is_on_gpu_; }

    /**
     * @brief Get initialization parameters
     * @return Vector of initialization parameters
     */
    const std::vector<double>& get_init_params() const { return init_params; }

    // ===== UTILITY METHODS =====
    
    /**
     * @brief Print vector information for debugging
     */
    void print_info() const {
        std::cout << "caVector Info:" << std::endl;
        std::cout << "  Size: " << size_ << std::endl;
        std::cout << "  On GPU: " << (is_on_gpu_ ? "Yes" : "No") << std::endl;
        std::cout << "  HOST memory: " << (host_data.size() > 0 ? "Allocated" : "Not allocated") << std::endl;
        std::cout << "  GPU memory: " << (gpu_buffer ? "Allocated" : "Not allocated") << std::endl;
    }

    /**
     * @brief Get memory usage information
     * @return String with memory usage details
     */
    std::string get_memory_info() const {
        size_t host_memory = host_data.size() * sizeof(T);
        size_t gpu_memory = gpu_buffer ? gpu_buffer->get_size_bytes() : 0;

        return "HOST: " + std::to_string(host_memory) + " bytes, " +
               "GPU: " + std::to_string(gpu_memory) + " bytes";
    }

    // ===== VISUALIZATION METHODS =====

    /**
     * @brief Convert vector data to Python list
     * @return Vector data as a string representation
     */
    std::string to_list_string() const {
        if (host_data.empty()) {
            return "[]";
        }

        std::string result = "[";
        for (size_t i = 0; i < host_data.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(host_data[i]);
        }
        result += "]";
        return result;
    }

    /**
     * @brief Get first n elements as string
     * @param n Number of elements to show
     * @return String representation of first n elements
     */
    std::string head_string(size_t n = 5) const {
        if (host_data.empty()) {
            return "[]";
        }

        size_t count = std::min(n, host_data.size());
        std::string result = "[";
        for (size_t i = 0; i < count; ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(host_data[i]);
        }
        result += "]";
        return result;
    }

    /**
     * @brief Get last n elements as string
     * @param n Number of elements to show
     * @return String representation of last n elements
     */
    std::string tail_string(size_t n = 5) const {
        if (host_data.empty()) {
            return "[]";
        }

        size_t count = std::min(n, host_data.size());
        size_t start = host_data.size() - count;
        
        std::string result = "[";
        for (size_t i = start; i < host_data.size(); ++i) {
            if (i > start) result += ", ";
            result += std::to_string(host_data[i]);
        }
        result += "]";
        return result;
    }

    /**
     * @brief Smart string representation like numpy
     * @return String representation of the vector
     */
    std::string smart_string() const {
        if (host_data.size() <= 10) {
            return to_list_string() ;
        } else {
            std::string first = head_string(5);
            std::string last = tail_string(5);
            return first + " ... " + last + ", size=" + std::to_string(size_);
        }
    }

    // ===== COPY CONTROL =====

    /**
     * @brief Copy constructor
     * @param other Vector to copy from
     */
    caVector(const caVector& other)
        : size_(other.size_)
        , is_on_gpu_(false)  // New copy starts on HOST
        , init_function(other.init_function)
        , init_params(other.init_params) {

        // Copy HOST data
        host_data = other.host_data;

        // Note: GPU data is not copied (starts fresh on HOST)
    }

    /**
     * @brief Copy assignment operator
     * @param other Vector to copy from
     * @return Reference to this vector
     */
    caVector& operator=(const caVector& other) {
        if (this != &other) {
            size_ = other.size_;
            is_on_gpu_ = false;  // Reset to HOST
            init_function = other.init_function;
            init_params = other.init_params;

            // Copy HOST data
            host_data = other.host_data;

            // Reset GPU buffer (will be reallocated when needed)
            gpu_buffer.reset();
        }
        return *this;
    }

    /**
     * @brief Move constructor
     * @param other Vector to move from
     */
    caVector(caVector&& other) noexcept
        : host_data(std::move(other.host_data))
        , gpu_buffer(std::move(other.gpu_buffer))
        , size_(other.size_)
        , is_on_gpu_(other.is_on_gpu_)
        , init_function(std::move(other.init_function))
        , init_params(std::move(other.init_params)) {

        // Reset other vector
        other.size_ = 0;
        other.is_on_gpu_ = false;
    }

    /**
     * @brief Move assignment operator
     * @param other Vector to move from
     * @return Reference to this vector
     */
    caVector& operator=(caVector&& other) noexcept {
        if (this != &other) {
            host_data = std::move(other.host_data);
            gpu_buffer = std::move(other.gpu_buffer);
            size_ = other.size_;
            is_on_gpu_ = other.is_on_gpu_;
            init_function = std::move(other.init_function);
            init_params = std::move(other.init_params);

            // Reset other vector
            other.size_ = 0;
            other.is_on_gpu_ = false;
        }
        return *this;
    }
};

#endif // CA_VECTOR_CUH
