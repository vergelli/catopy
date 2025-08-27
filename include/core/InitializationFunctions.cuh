#ifndef INITIALIZATION_FUNCTIONS_CUH
#define INITIALIZATION_FUNCTIONS_CUH

/**
 * @file InitializationFunctions.cuh
 * @brief Predefined initialization functions for data structures
 *
 * This file provides a collection of commonly used initialization functions
 * that users can pass to caVector, caMatrix, and caTensor constructors.
 *
 * Usage examples:
 * @code
 * // Create vector with zeros
 * caVector<double> vec1(1000, ca::zeros<double>);
 *
 * // Create matrix with random values (seed=42)
 * caMatrix<double> mat(4, 4, ca::random<double>(42));
 *
 * // Create tensor with normal distribution (mean=0, std=1)
 * caTensor<double> ten(3, 3, 3, ca::normal<double>(0.0, 1.0));
 * @endcode
 */

#include <random>
#include <functional>
#include <vector>
#include <cmath>

namespace ca {

    // ===== BASIC INITIALIZATION FUNCTIONS =====
    
    /**
     * @brief Initialize data with zeros
     * @tparam T Data type
     * @return Function that sets all elements to zero
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> zeros() {
        return [](T* data, size_t size, const std::vector<double>& params) {
            (void)params; // Unused parameter
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<T>(0);
            }
        };
    }

    /**
     * @brief Initialize data with ones
     * @tparam T Data type
     * @return Function that sets all elements to one
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> ones() {
        return [](T* data, size_t size, const std::vector<double>& params) {
            (void)params; // Unused parameter
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<T>(1);
            }
        };
    }

    /**
     * @brief Initialize data with a constant value
     * @tparam T Data type
     * @param value Constant value to fill the data
     * @return Function that sets all elements to the specified value
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> constant(T value) {
        return [value](T* data, size_t size, const std::vector<double>& params) {
            (void)params; // Unused parameter
            for (size_t i = 0; i < size; ++i) {
                data[i] = value;
            }
        };
    }

    // ===== RANDOM INITIALIZATION FUNCTIONS =====
    
    /**
     * @brief Initialize data with uniform random values between 0 and 1
     * @tparam T Data type
     * @param seed Random seed (optional, uses random_device if not provided)
     * @return Function that fills data with uniform random values
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> random(int seed = -1) {
        return [seed](T* data, size_t size, const std::vector<double>& params) {
            // Use provided seed or generate random one
            std::random_device rd;
            std::mt19937 gen(seed >= 0 ? static_cast<unsigned int>(seed) : rd());
            std::uniform_real_distribution<T> dis(0.0, 1.0);
            
            for (size_t i = 0; i < size; ++i) {
                data[i] = dis(gen);
            }
        };
    }

    /**
     * @brief Initialize data with uniform random values in specified range
     * @tparam T Data type
     * @param min Minimum value (default: 0.0)
     * @param max Maximum value (default: 1.0)
     * @param seed Random seed (optional)
     * @return Function that fills data with uniform random values in [min, max]
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> uniform(T min = 0.0, T max = 1.0, int seed = -1) {
        return [min, max, seed](T* data, size_t size, const std::vector<double>& params) {
            // Use provided seed or generate random one
            std::random_device rd;
            std::mt19937 gen(seed >= 0 ? static_cast<unsigned int>(seed) : rd());
            std::uniform_real_distribution<T> dis(min, max);
            
            for (size_t i = 0; i < size; ++i) {
                data[i] = dis(gen);
            }
        };
    }

    /**
     * @brief Initialize data with normal (Gaussian) distribution
     * @tparam T Data type
     * @param mean Mean of the distribution (default: 0.0)
     * @param std Standard deviation (default: 1.0)
     * @param seed Random seed (optional)
     * @return Function that fills data with normally distributed values
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> normal(T mean = 0.0, T std = 1.0, int seed = -1) {
        return [mean, std, seed](T* data, size_t size, const std::vector<double>& params) {
            // Use provided seed or generate random one
            std::random_device rd;
            std::mt19937 gen(seed >= 0 ? static_cast<unsigned int>(seed) : rd());
            std::normal_distribution<T> dis(mean, std);
            
            for (size_t i = 0; i < size; ++i) {
                data[i] = dis(gen);
            }
        };
    }

    /**
     * @brief Initialize data with Box-Muller transformation (normal distribution)
     * @tparam T Data type
     * @param mean Mean of the distribution (default: 0.0)
     * @param std Standard deviation (default: 1.0)
     * @param seed Random seed (optional)
     * @return Function that fills data using Box-Muller transformation
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> box_muller(T mean = 0.0, T std = 1.0, int seed = -1) {
        return [mean, std, seed](T* data, size_t size, const std::vector<double>& params) {
            // Use provided seed or generate random one
            std::random_device rd;
            std::mt19937 gen(seed >= 0 ? static_cast<unsigned int>(seed) : rd());
            std::uniform_real_distribution<T> dis(0.0, 1.0);
            
            for (size_t i = 0; i < size; i += 2) {
                // Box-Muller transformation
                T u1 = dis(gen);
                T u2 = dis(gen);
                
                // Avoid log(0) by ensuring u1 > 0
                if (u1 <= 0.0) u1 = std::numeric_limits<T>::epsilon();
                
                T z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
                T z1 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);
                
                data[i] = mean + std * z0;
                
                // Handle odd-sized arrays
                if (i + 1 < size) {
                    data[i + 1] = mean + std * z1;
                }
            }
        };
    }

    // ===== SEQUENTIAL INITIALIZATION FUNCTIONS =====
    
    /**
     * @brief Initialize data with sequential values starting from start
     * @tparam T Data type
     * @param start Starting value (default: 0)
     * @param step Step between consecutive values (default: 1)
     * @return Function that fills data with sequential values
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> sequence(T start = 0, T step = 1) {
        return [start, step](T* data, size_t size, const std::vector<double>& params) {
            (void)params; // Unused parameter
            for (size_t i = 0; i < size; ++i) {
                data[i] = start + static_cast<T>(i) * step;
            }
        };
    }

    /**
     * @brief Initialize data with values from 0 to size-1
     * @tparam T Data type
     * @return Function that fills data with values [0, 1, 2, ..., size-1]
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> arange() {
        return [](T* data, size_t size, const std::vector<double>& params) {
            (void)params; // Unused parameter
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<T>(i);
            }
        };
    }

    // ===== ADVANCED INITIALIZATION FUNCTIONS =====
    
    /**
     * @brief Initialize data with values from a mathematical function
     * @tparam T Data type
     * @param func Mathematical function (e.g., sin, cos, exp)
     * @param scale Scaling factor (default: 1.0)
     * @param offset Offset value (default: 0.0)
     * @return Function that fills data using the specified mathematical function
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> mathematical(std::function<T(T)> func, T scale = 1.0, T offset = 0.0) {
        return [func, scale, offset](T* data, size_t size, const std::vector<double>& params) {
            (void)params; // Unused parameter
            for (size_t i = 0; i < size; ++i) {
                T x = static_cast<T>(i);
                data[i] = scale * func(x) + offset;
            }
        };
    }

    /**
     * @brief Initialize data with sine wave pattern
     * @tparam T Data type
     * @param frequency Frequency of the sine wave (default: 1.0)
     * @param amplitude Amplitude of the sine wave (default: 1.0)
     * @param phase Phase shift in radians (default: 0.0)
     * @return Function that fills data with sine wave pattern
     */
    template<typename T>
    std::function<void(T*, size_t, const std::vector<double>&)> sine(T frequency = 1.0, T amplitude = 1.0, T phase = 0.0) {
        return [frequency, amplitude, phase](T* data, size_t size, const std::vector<double>& params) {
            (void)params; // Unused parameter
            for (size_t i = 0; i < size; ++i) {
                T x = static_cast<T>(i);
                data[i] = amplitude * std::sin(2.0 * M_PI * frequency * x + phase);
            }
        };
    }

} // namespace ca

#endif // INITIALIZATION_FUNCTIONS_CUH
