#ifndef GPU_BUFFER_CUH
#define GPU_BUFFER_CUH

/**
 * @file GpuBuffer.cuh
 * @brief Clase para gestión automática y segura de memoria GPU
 * 
 * Esta clase implementa el patrón RAII (Resource Acquisition Is Initialization)
 * para manejar memoria de la GPU de manera automática y segura.
 * 
 * Características principales:
 * - Asignación automática de memoria GPU al crear el objeto
 * - Liberación automática de memoria al destruir el objeto
 * - Prevención de memory leaks y double-free
 * - Soporte para move semantics (transferencia de propiedad)
 * - Prohibición de copias accidentales
 */

#include "../cuda_errors.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include "../../logger/Logger.cuh"

template<typename T>
class GpuBuffer {
private:
    T* gpu_pointer = nullptr;        // Puntero a memoria GPU
    size_t element_count = 0;         // Número de elementos almacenados

public:
    /**
     * @brief Constructor: asigna memoria GPU automáticamente
     * @param count Número de elementos de tipo T a almacenar
     * 
     * Ejemplo:
     * @code
     * GpuBuffer<double> buffer(1000);  // Almacena 1000 doubles en GPU
     * @endcode
     */
    GpuBuffer(size_t count) : element_count(count) {
        Logger::debug("GpuBuffer constructor called with count= {}", count);
        Logger::debug("sizeof(T) =  {}", sizeof(T));
        Logger::debug("Total bytes to allocate =  {}", (count * sizeof(T)));
        
        if (count == 0) {
            Logger::debug("ERROR - count is 0");
            throw std::invalid_argument("GpuBuffer: count must be greater than 0");
        }

        Logger::debug("About to call cudaMalloc...");
        
        // Asignar memoria en GPU usando CUDA
        try {
            CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_pointer, count * sizeof(T)));
            Logger::debug("cudaMalloc completed successfully");
        } catch (const std::exception& e) {
            Logger::debug("ERROR - cudaMalloc failed:  {}", e.what());
            throw;
        }

        Logger::debug("gpu_pointer =  {}", static_cast<void*>(gpu_pointer));
        
        if (gpu_pointer == nullptr) {
            Logger::debug("ERROR - gpu_pointer is null after cudaMalloc");
            throw std::runtime_error("GpuBuffer: Failed to allocate GPU memory");
        }
        
        Logger::debug("GpuBuffer constructor completed successfully");
    }

    /**
     * @brief Destructor: libera memoria GPU automáticamente
     * 
     * No necesitas llamar esto manualmente. Se ejecuta automáticamente
     * cuando el objeto sale de scope.
     */
    ~GpuBuffer() {
        Logger::debug("GpuBuffer destructor called");
        Logger::debug("gpu_pointer =  {}", static_cast<void*>(gpu_pointer));
        Logger::debug("element_count =  {}", element_count);
        
        if (gpu_pointer != nullptr) {
            Logger::debug("About to call cudaFree on gpu_pointer =  {}", static_cast<void*>(gpu_pointer));
            try {
                CHECK_CUDA_ERROR(cudaFree(gpu_pointer));
                Logger::debug("cudaFree completed successfully");
            } catch (const std::exception& e) {
                Logger::debug("ERROR - cudaFree failed:  {}", e.what());
            } catch (...) {
                Logger::debug("ERROR - Unknown exception in cudaFree");
            }
            gpu_pointer = nullptr;
        } else {
            Logger::debug("gpu_pointer was already nullptr in destructor");
        }
        
        Logger::debug("About to exit GpuBuffer destructor");
        Logger::debug("GpuBuffer destructor completed");
    }

    /**
     * @brief Obtener puntero a memoria GPU
     * @return Puntero raw a la memoria GPU

     * Usar este puntero para operaciones CUDA.
     * NO liberar manualmente - GpuBuffer se encarga de eso.
     */
    T* get_pointer() const { 
        return gpu_pointer; 
    }

    /**
     * @brief Obtener número de elementos almacenados
     * @return Número de elementos de tipo T
     */
    size_t get_size() const { 
        return element_count; 
    }

    /**
     * @brief Obtener tamaño total en bytes
     * @return Tamaño total de memoria en bytes
     */
    size_t get_size_bytes() const { 
        return element_count * sizeof(T); 
    }

    /**
     * @brief Verificar si el buffer está vacío
     * @return true si no hay elementos, false en caso contrario
     */
    bool is_empty() const { 
        return element_count == 0; 
    }

    // ===== PROHIBICIÓN DE COPIAS =====
    // No permitir copias accidentales que podrían causar double-free

    /**
     * @brief Constructor de copia PROHIBIDO
     *
     * No se pueden copiar buffers GPU porque cada uno debe tener
     * su propia memoria. Copiar podría causar double-free.
     */
    GpuBuffer(const GpuBuffer&) = delete;

    /**
     * @brief Operador de asignación por copia PROHIBIDO
     *
     * Misma razón: no permitir copias accidentales.
     */
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // ===== SOPORTE PARA MOVE SEMANTICS =====
    // Permitir transferir propiedad de un buffer a otro

    /**
     * @brief Constructor de movimiento: transfiere propiedad
     * @param other Buffer del cual tomar la propiedad
     * 
     * Después del movimiento, 'other' queda en estado inválido
     * y no debe usarse.
     */
    GpuBuffer(GpuBuffer&& other) noexcept
        : gpu_pointer(other.gpu_pointer), element_count(other.element_count) {
        // Transferir propiedad
        other.gpu_pointer = nullptr;
        other.element_count = 0;
    }

    /**
     * @brief Operador de asignación por movimiento: transfiere propiedad
     * @param other Buffer del cual tomar la propiedad
     * @return Referencia a este buffer
     *
     * Libera la memoria actual antes de tomar la nueva.
     */
    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            // Liberar memoria actual
            if (gpu_pointer != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(gpu_pointer));
            }

            // Transferir propiedad
            gpu_pointer = other.gpu_pointer;
            element_count = other.element_count;

            // Invalidar 'other'
            other.gpu_pointer = nullptr;
            other.element_count = 0;
        }
        return *this;
    }
};

#endif // GPU_BUFFER_CUH
