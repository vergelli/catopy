#ifndef CA_VECTOR_CUH
#define CA_VECTOR_CUH

#include "../device/memory/GpuBuffer.cuh"
#include "../device/memory/MemoryTransfer.cuh"
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
    
    // Dirty flags for memory consistency
    bool host_dirty_;                                           // HOST data has been modified
    bool gpu_dirty_;                                            // GPU data has been modified

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
        , host_dirty_(false)
        , gpu_dirty_(false)
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
        , host_dirty_(false)
        , gpu_dirty_(false)
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
                for (size_t i = 0; i < std::min(size, size_t(5)); ++i) {
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
     * This method implements the "lazy copy" pattern with dirty flags:
     * - If data is already on GPU and not dirty, does nothing
     * - If data is only on HOST, allocates GPU memory and transfers data
     * - If HOST data was modified (host_dirty_), transfers HOST to GPU
     * - Sets is_on_gpu_ flag to true and clears dirty flags
     */
    void ensure_on_gpu() {
        if (!is_on_gpu_) {
            // First time: allocate GPU memory and transfer data
            gpu_buffer = std::make_unique<GpuBuffer<T>>(size_);

            // Transfer data from HOST to GPU using our MemoryTransfer namespace
            MemoryTransfer::host_to_gpu(
                gpu_buffer->get_pointer(),
                host_data.data(),
                size_
            );

            is_on_gpu_ = true;
            host_dirty_ = false;
            gpu_dirty_ = false;
        } else if (host_dirty_) {
            // HOST was modified: transfer HOST to GPU to sync
            MemoryTransfer::host_to_gpu(
                gpu_buffer->get_pointer(),
                host_data.data(),
                size_
            );

            host_dirty_ = false;
            gpu_dirty_ = false;
        } else if (gpu_dirty_) {
            // GPU was marked as dirty: HOST data is correct, just clear flags
            // No need to transfer HOST → GPU since HOST has the latest data
            host_dirty_ = false;
            gpu_dirty_ = false;
        }
        // If all flags are false, no action needed (GPU is already up-to-date)
    }

    /**
     * @brief Ensures data is available on HOST
     * 
     * If data was modified on GPU (gpu_dirty_), this method transfers it back to HOST
     * to ensure consistency.
     */
private:
    void sync_gpu_to_host() {
        if (gpu_dirty_) {
            // GPU was modified: transfer GPU to HOST to sync
            MemoryTransfer::gpu_to_host(
                host_data.data(),
                gpu_buffer->get_pointer(),
                size_
            );
            
            gpu_dirty_ = false;
            host_dirty_ = false;
        }
    }

public:
    void ensure_on_host() {
        sync_gpu_to_host();
        // After ensure_on_host(), data is guaranteed to be on HOST
        // Mark as not on GPU (even if GPU buffer exists)
        is_on_gpu_ = false;
    }

    // ===== DATA ACCESS =====

    /**
     * @brief Get pointer to HOST data
     * @return Raw pointer to HOST memory
     * 
     * Use this for CPU operations. Data is guaranteed to be on HOST.
     */
    T* host_ptr() {
        ensure_on_host();
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

    // ===== INDEXING OPERATORS =====

    /**
     * @brief Get element at specified index (const version)
     * @param index Index of the element to retrieve
     * @return Const reference to the element
     * 
     * This operator ensures data is on HOST before accessing.
     * Throws std::out_of_range if index is invalid.
     */
    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("caVector: index out of range");
        }
        // For const version, we can't call ensure_on_host() as it modifies state
        // We assume HOST data is available and up-to-date
        return host_data[index];
    }

    /**
     * @brief Get element at specified index (non-const version)
     * @param index Index of the element to retrieve
     * @return Reference to the element
     * 
     * This operator ensures data is on HOST before accessing.
     * Only marks GPU as dirty if GPU exists.
     * Throws std::out_of_range if index is invalid.
     */
    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("caVector: index out of range");
        }

        // Don't call sync_gpu_to_host when modifying HOST
        // HOST data is correct and doesn't need sync from GPU

        // Mark GPU as dirty if GPU exists
        if (is_on_gpu_) {
            gpu_dirty_ = true; // Mark GPU as dirty (will be synced when needed)
        }

        return host_data[index];
    }

    /**
     * @brief Get element at specified index with bounds checking
     * @param index Index of the element to retrieve
     * @return Reference to the element
     * 
     * This method provides bounds checking and is safer than operator[].
     * Throws std::out_of_range if index is invalid.
     */
    T& at(size_t index) {
        if (index >= size_) {
            throw std::out_of_range("caVector: index " + std::to_string(index) + " out of range (size: " + std::to_string(size_) + ")");
        }

        // Don't call sync_gpu_to_host when modifying HOST
        // HOST data is correct and doesn't need sync from GPU
        
        // Only mark GPU as dirty if GPU exists
        if (is_on_gpu_) {
            gpu_dirty_ = true; // Mark GPU as dirty
        }

        return host_data[index];
    }

    /**
     * @brief Get element at specified index with bounds checking (const version)
     * @param index Index of the element to retrieve
     * @return Const reference to the element
     * 
     * This method provides bounds checking and is safer than operator[].
     * Throws std::out_of_range if index is invalid.
     */
    const T& at(size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("caVector: index " + std::to_string(index) + " out of range (size: " + std::to_string(index) + ")");
        }
        // For const version, we can't call ensure_on_host() as it modifies state
        // We assume HOST data is available and up-to-date
        return host_data[index];
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
     * @brief Check if HOST data has been modified
     * @return true if HOST data is dirty, false if clean
     */
    bool is_host_dirty() const { return host_dirty_; }

    /**
     * @brief Check if GPU data has been modified
     * @return true if GPU data is dirty, false if clean
     */
    bool is_gpu_dirty() const { return gpu_dirty_; }

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
        std::cout << "  HOST dirty: " << (host_dirty_ ? "Yes" : "No") << std::endl;
        std::cout << "  GPU dirty: " << (gpu_dirty_ ? "Yes" : "No") << std::endl;
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
     * @brief Smart string representation
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

    // ===== SLICING OPERATIONS =====

    /**
     * @brief Create a slice of the vector
     * @param start Starting index (inclusive)
     * @param stop Ending index (exclusive)
     * @param step Step size (default: 1)
     * @return New caVector containing the sliced data
     * 
     * This method creates a new vector with data from the specified range.
     * The new vector is completely independent (deep copy).
     * 
     * Examples:
     * - slice(1, 5) → elements [1, 2, 3, 4]
     * - slice(0, 10, 2) → elements [0, 2, 4, 6, 8]
     * - slice(5, 0, -1) → elements [5, 4, 3, 2, 1] (reverse)
     */
    caVector<T> slice(size_t start, size_t stop, size_t step = 1) const {
        // Validate parameters
        if (step == 0) {
            throw std::invalid_argument("caVector::slice: step cannot be zero");
        }

        // Handle negative indices (Python-style)
        if (start >= size_) start = size_;
        if (stop > size_) stop = size_;

        // Handle reverse slicing
        if (step < 0) {
            if (start >= size_) start = size_ - 1;
            if (stop > size_) stop = size_;
            if (start < stop) {
                // Swap start and stop for reverse slicing
                std::swap(start, stop);
            }
        }

        // Calculate slice size
        size_t slice_size = 0;
        if (step > 0) {
            if (start < stop) {
                slice_size = (stop - start + step - 1) / step;
            }
        } else {
            if (start > stop) {
                slice_size = (start - stop - step - 1) / (-step);
            }
        }

        if (slice_size == 0) {
            // Return empty vector
            return caVector<T>(0, [](T*, size_t, const std::vector<double>&) {}, {});
        }

        // Create new vector with slice size
        caVector<T> result(slice_size, [](T*, size_t, const std::vector<double>&) {}, {});

        // Copy data from slice
        size_t src_idx = start;
        size_t dst_idx = 0;

        if (step > 0) {
            // Forward slicing
            while (src_idx < stop && dst_idx < slice_size) {
                result.host_data[dst_idx] = host_data[src_idx];
                src_idx += step;
                dst_idx++;
            }
        } else {
            // Reverse slicing
            while (src_idx > stop && dst_idx < slice_size) {
                result.host_data[dst_idx] = host_data[src_idx];
                src_idx += step;  // step is negative
                dst_idx++;
            }
        }

        // Ensure the result vector has the correct size
        result.size_ = dst_idx;
        result.host_data.resize(dst_idx);

        return result;
    }

    /**
     * @brief Create a slice using a range (start:stop)
     * @param range Pair of start and stop indices
     * @return New caVector containing the sliced data
     * 
     * This is a convenience method for common slicing operations.
     * Equivalent to slice(range.first, range.second, 1)
     */
    caVector<T> slice(const std::pair<size_t, size_t>& range) const {
        return slice(range.first, range.second, 1);
    }

    // ===== COPY CONTROL =====

    /**
     * @brief Copy constructor
     * @param other Vector to copy from
     */
    caVector(const caVector& other)
        : size_(other.size_)
        , is_on_gpu_(false)  // New copy starts on HOST
        , host_dirty_(false)  // New copy starts clean
        , gpu_dirty_(false)   // New copy starts clean
        , init_function(other.init_function)
        , init_params(other.init_params) {

        // Copy HOST data
        host_data = other.host_data;

        // Note: GPU data is not copied (starts fresh on HOST)
        // Dirty flags are reset for the new copy
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
            host_dirty_ = false;  // Reset dirty flags
            gpu_dirty_ = false;   // Reset dirty flags
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
        , host_dirty_(other.host_dirty_)
        , gpu_dirty_(other.gpu_dirty_)
        , init_function(std::move(other.init_function))
        , init_params(std::move(other.init_params)) {

        // Reset other vector
        other.size_ = 0;
        other.is_on_gpu_ = false;
        other.host_dirty_ = false;
        other.gpu_dirty_ = false;
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
            host_dirty_ = other.host_dirty_;
            gpu_dirty_ = other.gpu_dirty_;
            init_function = std::move(other.init_function);
            init_params = std::move(other.init_params);

            // Reset other vector
            other.size_ = 0;
            other.is_on_gpu_ = false;
            other.host_dirty_ = false;
            other.gpu_dirty_ = false;
        }
        return *this;
    }
};

#endif // CA_VECTOR_CUH
